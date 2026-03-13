import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path("/home/dataset-local/lr/code/fusionnext")
BEVDET_ROOT = Path("/home/dataset-local/lr/code/BEVDet")
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "fusionnext_nuscenes_mini_3d.py"


def ensure_pythonpath() -> None:
    for path in (str(PROJECT_ROOT), str(BEVDET_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FusionNeXt training helper")
    parser.add_argument(
        "--mode",
        choices=["smoke", "train"],
        default="smoke",
        help="smoke: build dataset/model and run one training forward; train: delegate to BEVDet tools/train.py",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="OpenMMLab config file path",
    )
    parser.add_argument("--work-dir", default=None, help="Override work_dir in config")
    parser.add_argument("--gpu-id", type=int, default=0, help="CUDA device id")
    parser.add_argument(
        "--cfg-options",
        nargs="*",
        default=None,
        help="Config overrides in key=value format, same as OpenMMLab train.py",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation during training when mode=train",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples for smoke batch collation",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Dataset index to inspect in smoke mode",
    )
    parser.add_argument(
        "--backward",
        action="store_true",
        help="Also run backward() in smoke mode",
    )
    return parser.parse_args()


def load_cfg(config_path: str, cfg_options) -> "Config":
    ensure_pythonpath()
    from mmcv import Config
    from mmcv.utils import import_modules_from_strings

    cfg = Config.fromfile(config_path)
    if cfg.get("custom_imports"):
        import_modules_from_strings(**cfg["custom_imports"])
    if cfg_options:
        from mmcv import DictAction  # noqa: F401  # parity note for CLI semantics
        merge_dict = {}
        for item in cfg_options:
            key, value = item.split("=", 1)
            merge_dict[key] = eval(value) if value[:1] in "[{('\"" or value in {"True", "False", "None"} else value
        cfg.merge_from_dict(merge_dict)
    return cfg


def move_data_to_device(data, device):
    from mmcv.parallel import scatter

    return scatter(data, [device.index])[0]


def summarize_losses(losses, model):
    loss, log_vars = model._parse_losses(losses)
    scalar_logs = {
        key: float(value.detach().cpu()) if hasattr(value, "detach") else float(value)
        for key, value in log_vars.items()
    }
    return loss, scalar_logs


def run_smoke(args: argparse.Namespace) -> None:
    ensure_pythonpath()
    import torch
    from mmcv.parallel import collate
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model

    if not torch.cuda.is_available():
        raise RuntimeError("Smoke mode requires CUDA in the bevdet environment.")

    cfg = load_cfg(args.config, args.cfg_options)
    cfg.data.samples_per_gpu = args.num_samples
    cfg.data.workers_per_gpu = 0
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    dataset = build_dataset(cfg.data.train)
    samples = []
    for offset in range(args.num_samples):
        dataset_index = min(args.sample_index + offset, len(dataset) - 1)
        samples.append(dataset[dataset_index])

    batch = collate(samples, samples_per_gpu=args.num_samples)
    device = torch.device(f"cuda:{args.gpu_id}")
    batch = move_data_to_device(batch, device)

    model = build_model(
        cfg.model,
    )
    model.init_weights()
    model = model.to(device)
    model.train()

    losses = model(return_loss=True, **batch)
    total_loss, log_vars = summarize_losses(losses, model)

    print("Smoke test passed.")
    print(f"Dataset length: {len(dataset)}")
    print(f"Batch size: {args.num_samples}")
    print(f"Loss keys: {sorted(losses.keys())}")
    print(f"Log vars: {log_vars}")

    if args.backward:
        total_loss.backward()
        print("Backward passed.")


def run_train(args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        str(BEVDET_ROOT / "tools" / "train.py"),
        args.config,
        "--gpu-id",
        str(args.gpu_id),
    ]
    if args.work_dir is not None:
        command.extend(["--work-dir", args.work_dir])
    if args.validate:
        command.append("--validate")
    if args.cfg_options:
        command.extend(["--cfg-options", *args.cfg_options])

    env = os.environ.copy()
    pythonpath_parts = [str(PROJECT_ROOT), str(BEVDET_ROOT)]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    print("Running command:")
    print(" ".join(command))
    subprocess.run(command, cwd=str(BEVDET_ROOT), env=env, check=True)


def main() -> None:
    args = parse_args()
    if args.mode == "smoke":
        run_smoke(args)
    else:
        run_train(args)


if __name__ == "__main__":
    main()
