import os
import sys


PROJECT_ROOT = "/home/dataset-local/lr/code/fusionnext"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from nuscenes_pipeline import build_fusionnext_dataset_cfg, build_fusionnext_model_cfg


custom_imports = dict(
    imports=["fusion", "fusion_head", "nuscenes_pipeline"],
    allow_failed_imports=False,
)

plugin = False

data_root = "/home/dataset-local/lr/data/nuscenes/"
train_ann_file = os.path.join(data_root, "nuscenes_infos_train_mini_sweep.pkl")
val_ann_file = os.path.join(data_root, "nuscenes_infos_val_mini_sweep.pkl")

class_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
voxel_size = [0.2, 0.2, 0.4]

data_config = dict(
    cams=[
        "CAM_FRONT_LEFT",
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT",
        "CAM_BACK",
        "CAM_BACK_RIGHT",
    ],
    Ncams=6,
    input_size=(256, 704),
    src_size=(900, 1600),
    resize=(-0.06, 0.11),
    rot=(-5.4, 5.4),
    flip=True,
    crop_h=(0.0, 0.0),
    resize_test=0.0,
)

bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5,
)

bbox_head = dict(
    type="FusionNeXtSimple3DHead",
    num_classes=len(class_names),
    in_channels=256,
    hidden_channels=256,
    voxel_size=voxel_size,
    point_cloud_range=point_cloud_range,
    loss_cls_weight=1.0,
    loss_bbox_weight=2.0,
    loss_dir_weight=0.2,
    score_thr=0.1,
    nms_thr=0.2,
    max_num=100,
)

model_train_cfg = dict()
model_test_cfg = dict(score_thr=0.1)

model = build_fusionnext_model_cfg(
    point_cloud_range=point_cloud_range,
    voxel_size=voxel_size,
    embed_dim=256,
    image_weights="default",
    lidar_in_channels=5,
    bbox_head=bbox_head,
    train_cfg=model_train_cfg,
    test_cfg=model_test_cfg,
)

dataset_type = "NuScenesDataset"
file_client_args = dict(backend="disk")

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=build_fusionnext_dataset_cfg(
        ann_file=train_ann_file,
        data_root=data_root,
        data_config=data_config,
        class_names=class_names,
        point_cloud_range=point_cloud_range,
        bda_aug_conf=bda_aug_conf,
        test_mode=False,
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args,
    ),
    val=build_fusionnext_dataset_cfg(
        ann_file=val_ann_file,
        data_root=data_root,
        data_config=data_config,
        class_names=class_names,
        point_cloud_range=point_cloud_range,
        bda_aug_conf=bda_aug_conf,
        test_mode=True,
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args,
    ),
    test=build_fusionnext_dataset_cfg(
        ann_file=val_ann_file,
        data_root=data_root,
        data_config=data_config,
        class_names=class_names,
        point_cloud_range=point_cloud_range,
        bda_aug_conf=bda_aug_conf,
        test_mode=True,
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args,
    ),
)

optimizer = dict(
    type="AdamW",
    lr=2e-4,
    betas=(0.9, 0.999),
    weight_decay=0.01,
)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy="step",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=1.0 / 3.0,
    step=[8, 11],
)

runner = dict(type="EpochBasedRunner", max_epochs=12)

checkpoint_config = dict(interval=1)
log_config = dict(interval=20, hooks=[dict(type="TextLoggerHook")])
evaluation = dict(interval=1)

dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = None
resume_from = None
workflow = [("train", 1)]
work_dir = os.path.join(PROJECT_ROOT, "work_dirs", "fusionnext_nuscenes_mini_3d")
