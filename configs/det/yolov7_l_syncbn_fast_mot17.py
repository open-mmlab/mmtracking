_base_ = [
    '../_base_/default_runtime.py',
]

img_scale = (1088, 1088)
batch_size = 2
max_epochs = 5
save_epoch_intervals = 1

# different from yolov5
anchors = [
    [(12, 16), (19, 36), (40, 28)],  # P3/8
    [(36, 75), (76, 55), (72, 146)],  # P4/16
    [(142, 110), (192, 243), (459, 401)]  # P5/32
]
strides = [8, 16, 32]
num_det_layers = 3
num_classes = 1

# single-scale training is recommended to
# be turned on, which can speed up training.
env_cfg = dict(cudnn_benchmark=True)

model = dict(
    type='YOLODetector',
    _scope_='mmyolo',
    data_preprocessor=dict(
        type='YOLOv5DetDataPreprocessor',
        pad_size_divisor=32,
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
        bgr_to_rgb=True),
    backbone=dict(
        type='YOLOv7Backbone',
        arch='L',
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True)),
    neck=dict(
        type='YOLOv7PAFPN',
        block_cfg=dict(
            type='ELANBlock',
            middle_ratio=0.5,
            block_ratio=0.25,
            num_blocks=4,
            num_convs_in_block=1),
        upsample_feats_cat_first=False,
        in_channels=[512, 1024, 1024],
        # The real output channel will be multiplied by 2
        out_channels=[128, 256, 512],
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='YOLOv7Head',
        head_module=dict(
            type='YOLOv7HeadModule',
            num_classes=num_classes,
            in_channels=[256, 512, 1024],
            featmap_strides=strides,
            num_base_priors=3),
        prior_generator=dict(
            type='mmdet.YOLOAnchorGenerator',
            base_sizes=anchors,
            strides=strides),
        # scaled based on number of detection layers
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=0.3 * (num_classes / 80 * 3 / num_det_layers)),
        loss_bbox=dict(
            type='IoULoss',
            iou_mode='ciou',
            bbox_format='xywh',
            reduction='mean',
            loss_weight=0.05 * (3 / num_det_layers),
            return_iou=True),
        loss_obj=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=0.7 * ((img_scale[0] / 640)**2 * 3 / num_det_layers)),
        obj_level_weights=[4., 1., 0.4],
        # BatchYOLOv7Assigner params
        prior_match_thr=4.,
        simota_candidate_topk=10,
        simota_iou_weight=3.0,
        simota_cls_weight=1.0),
    init_cfg=dict(
        type='Pretrained',
        checkpoint='https://download.openmmlab.com/mmyolo/v0/yolov7/yolov7_l_syncbn_fast_8x16b-300e_coco/yolov7_l_syncbn_fast_8x16b-300e_coco_20221123_023601-8113c0eb.pth'),
    test_cfg=dict(
        multi_label=False,
        nms_pre=30000,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.65),
        max_per_img=300))

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='mmdet.Resize',
        scale=img_scale,
        keep_ratio=True,
        clip_object_border=False),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.Pad', size=img_scale, pad_val=0),
    dict(type='mmdet.PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
    dict(type='mmdet.Pad', size_divisor=32, pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

# dataset settings
dataset_type = 'mmdet.CocoDataset'
data_root = 'data/MOT17/'

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='mmdet.AspectRatioBatchSampler'),
    collate_fn=dict(type='yolov5_collate'),  # FASTER
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/half-train_cocoformat.json',
        data_prefix=dict(img='train/'),
        metainfo=dict(classes=('pedestrian', )),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        _scope_='mmdet',
        ann_file='annotations/half-val_cocoformat.json',
        data_prefix=dict(img='train/'),
        metainfo=dict(classes=('pedestrian', )),
        test_mode=True,
        pipeline=test_pipeline))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='mmdet.CocoMetric',
    ann_file=data_root + 'annotations/half-val_cocoformat.json',
    metric='bbox',
    format_only=False)
test_evaluator = val_evaluator

param_scheduler = None
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=1e-3,
        momentum=0.937,
        weight_decay=0.0005,
        nesterov=True))

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=save_epoch_intervals,
    dynamic_intervals=[(270, 1)])

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    param_scheduler=dict(
        type='YOLOv5ParamSchedulerHook',
        _scope_='mmyolo',
        scheduler_type='cosine',
        lr_factor=0.1,  # note
        max_epochs=max_epochs),
    checkpoint=dict(
        type='CheckpointHook',
        save_param_scheduler=False,
        interval=1,
        save_best='auto',
        max_keep_ckpts=3))

custom_hooks = [
    dict(
        type='mmdet.EMAHook',
        ema_type='mmdet.ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49)
]
