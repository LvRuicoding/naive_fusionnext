import os


default_scope = "mmdet3d"

custom_imports = dict(
    imports=[
        "fusionnext.datasets.fusionnext_nuscenes_dataset",
        "fusionnext.datasets.pipelines.fusionnext_nuscenes",
        "fusionnext.models.data_preprocessors.fusion_det3d_data_preprocessor",
        "fusionnext.models.detectors.fusionnext",
        "fusionnext.models.dense_heads.fusionnext_simple_head",
    ],
    allow_failed_imports=False,
)

data_root = os.environ.get("FUSIONNEXT_DATA_ROOT", "data/nuscenes/")
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
metainfo = dict(classes=class_names)

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

model = dict(
    type="FusionNeXt",
    data_preprocessor=dict(type="FusionDet3DDataPreprocessor"),
    point_cloud_range=point_cloud_range,
    voxel_size=voxel_size,
    embed_dim=256,
    image_weights="default",
    lidar_in_channels=5,
    bbox_head=bbox_head,
    train_cfg=dict(),
    test_cfg=dict(score_thr=0.1),
)

backend_args = None

train_pipeline = [
    dict(type="FusionNeXtPrepareImageInputs", is_train=True, data_config=data_config),
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
        backend_args=backend_args,
    ),
    dict(type="FusionNeXtLoadAnnotations", classes=class_names),
    dict(type="FusionNeXtObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="FusionNeXtObjectNameFilter", classes=class_names),
    dict(type="FusionNeXtPrepareMeta"),
    dict(
        type="PackFusionDetInputs",
        keys=["points", "img_inputs", "gt_bboxes_3d", "gt_labels_3d"],
        meta_keys=("sample_idx", "token", "timestamp", "lidar_path", "box_type_3d", "box_mode_3d", "fusionnext_meta"),
    ),
]

test_pipeline = [
    dict(type="FusionNeXtPrepareImageInputs", data_config=data_config),
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
        backend_args=backend_args,
    ),
    dict(type="FusionNeXtPrepareMeta"),
    dict(
        type="PackFusionDetInputs",
        keys=["points", "img_inputs"],
        meta_keys=("sample_idx", "token", "timestamp", "lidar_path", "box_type_3d", "box_mode_3d", "fusionnext_meta"),
    ),
]

dataset_common = dict(
    type="FusionNuScenesDataset",
    data_root=data_root,
    metainfo=metainfo,
    modality=dict(
        use_lidar=True,
        use_camera=True,
        use_radar=False,
        use_map=False,
        use_external=False,
    ),
    box_type_3d="LiDAR",
)

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    collate_fn=dict(type="pseudo_collate"),
    dataset=dict(
        **dataset_common,
        ann_file=train_ann_file,
        pipeline=train_pipeline,
        test_mode=False,
    ),
)

val_dataloader = None
test_dataloader = None

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=12, val_interval=1)
val_cfg = None
test_cfg = None
val_evaluator = None
test_evaluator = None

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(
        type="AdamW",
        lr=2e-4,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    ),
    clip_grad=dict(max_norm=35, norm_type=2),
)

param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=1.0 / 3.0,
        by_epoch=False,
        begin=0,
        end=500,
    ),
    dict(
        type="MultiStepLR",
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1,
    ),
]

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=20),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", interval=1),
    sampler_seed=dict(type="DistSamplerSeedHook"),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)

log_processor = dict(type="LogProcessor", window_size=20, by_epoch=True)
log_level = "INFO"
load_from = None
resume = False
work_dir = os.environ.get("FUSIONNEXT_WORK_DIR", "./work_dirs/fusionnext_nuscenes_mini_3d")
