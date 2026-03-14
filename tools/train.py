import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "fusionnext_nuscenes_mini_3d.py"


def ensure_pythonpath() -> None:
    for path in (str(PROJECT_ROOT),):
        if path not in sys.path:
            sys.path.insert(0, path)


def parse_args() -> argparse.Namespace:
    from mmengine.config import DictAction

    parser = argparse.ArgumentParser(description="FusionNeXt training helper")
    parser.add_argument(
        "--mode",
        choices=["smoke", "train"],
        default="smoke",
        help="smoke: build dataset/model and run one training forward; train: run mmengine Runner",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Config file path",
    )
    parser.add_argument("--work-dir", default=None, help="Override work_dir in config")
    parser.add_argument("--gpu-id", type=int, default=0, help="CUDA device id for smoke mode")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        default=None,
        help="Config overrides in key=value format",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Enable validation loop if the config defines a validator",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="Job launcher",
    )
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
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
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


def load_cfg(config_path: str, cfg_options) -> "Config":
    ensure_pythonpath()
    from mmengine.config import Config
    from mmengine.utils import import_modules_from_strings

    cfg = Config.fromfile(config_path)
    if cfg.get("custom_imports"):
        import_modules_from_strings(**cfg["custom_imports"])
    if cfg_options:
        cfg.merge_from_dict(cfg_options)
    return cfg


def summarize_losses(losses):
    scalar_logs = {}
    total_loss = losses.get("loss")
    if total_loss is None:
        total_loss = sum(value for key, value in losses.items() if key.startswith("loss"))

    for key, value in losses.items():
        scalar_logs[key] = float(value.detach().cpu()) if hasattr(value, "detach") else float(value)
    return total_loss, scalar_logs


def run_smoke(args: argparse.Namespace) -> None:
    ensure_pythonpath()
    import torch
    from mmengine.dataset import pseudo_collate
    from mmengine.registry import init_default_scope
    from mmdet3d.registry import DATASETS, MODELS

    if not torch.cuda.is_available():
        raise RuntimeError("Smoke mode requires CUDA in the fusion environment.")

    cfg = load_cfg(args.config, args.cfg_options)
    if cfg.get("default_scope"):
        init_default_scope(cfg.default_scope)
    cfg.train_dataloader.num_workers = 0
    cfg.train_dataloader.batch_size = args.num_samples
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    samples = []
    for offset in range(args.num_samples):
        dataset_index = min(args.sample_index + offset, len(dataset) - 1)
        samples.append(dataset[dataset_index])

    batch = pseudo_collate(samples)
    device = torch.device(f"cuda:{args.gpu_id}")

    model = MODELS.build(cfg.model)
    model.init_weights()
    model = model.to(device)
    model.train()

    processed = model.data_preprocessor(batch, training=True)
    losses = model(**processed, mode="loss")
    total_loss, log_vars = summarize_losses(losses)

    print("Smoke test passed.")
    print(f"Dataset length: {len(dataset)}")
    print(f"Batch size: {args.num_samples}")
    print(f"Loss keys: {sorted(losses.keys())}")
    print(f"Log vars: {log_vars}")

    if args.backward:
        total_loss.backward()
        print("Backward passed.")


def run_train(args: argparse.Namespace) -> None:
    from mmengine.registry import init_default_scope
    from mmengine.runner import Runner

    cfg = load_cfg(args.config, args.cfg_options)
    if cfg.get("default_scope"):
        init_default_scope(cfg.default_scope)
    cfg.launcher = args.launcher

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    if args.validate:
        if cfg.get("val_cfg") is None or cfg.get("val_evaluator") is None:
            raise RuntimeError("This config does not define a validation loop or evaluator.")
    else:
        cfg.val_cfg = None
        cfg.val_dataloader = None
        cfg.val_evaluator = None

    runner = Runner.from_cfg(cfg)
    runner.train()


def main() -> None:
    args = parse_args()
    if args.mode == "smoke":
        run_smoke(args)
    else:
        run_train(args)


if __name__ == "__main__":
    main()
