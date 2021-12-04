_base_ = [
    '../../_base_/models/yolox_x_8x8.py',
    '../../_base_/datasets/mot_challenge.py', '../../_base_/default_runtime.py'
]
model = dict(
    type='ByteTrack',
    detector=dict(
        bbox_head=dict(num_classes=1),
        test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.7)),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            # '../convert_bytetrack_ablation.pth.tar'  # noqa: E501
            '../convert_legacy_yolox_x.pth.tar'  # noqa: E501
        )),
    motion=dict(type='KalmanFilter'),
    tracker=dict(
        type='ByteTracker',
        high_det_score=0.6,
        low_det_score=0.1,
        init_track_score=0.7,
        weight_iou=True,
        first_match_iou_thr=0.1,
        num_frames_retain=30))

data_root = 'data/MOT17/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_scale = (800, 1440)

train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(
        type='MixUp',
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', keep_ratio=True),
    dict(type='Pad', pad_to_square=True, pad_val=114.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Pad', size=img_scale, pad_val=114.0),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='VideoCollect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=4,
    train=dict(
        _delete_=True,
        type='MultiImageMixDataset',
        dataset=dict(
            type='CocoDataset',
            ann_file=[
                data_root + 'annotations/half-train_cocoformat.json',
                'data/crowdhuman/annotations/crowdhuman_train.json',
                'data/crowdhuman/annotations/crowdhuman_val.json'
            ],
            img_prefix=[
                data_root + 'train', 'data/crowdhuman/train',
                'data/crowdhuman/val'
            ],
            classes=('pedestrian', ),
            pipeline=[
                dict(type='LoadImageFromFile', to_float32=True),
                dict(type='LoadAnnotations', with_bbox=True)
            ],
            filter_empty_gt=False),
        pipeline=train_pipeline,
        dynamic_scale=img_scale),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))

# optimizer
# default 8 gpu
optimizer = dict(
    type='SGD',
    lr=0.001 / 8 * 3,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0))

# some hyper parameters
interval = 5
num_last_epochs = 10
total_epochs = 80
resume_from = None
ratio_range = (18, 32)

# learning policy
lr_config = dict(
    policy='YOLOX',
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,
    warmup_iters=1,
    num_last_epochs=num_last_epochs,
    min_lr_ratio=0.05)

custom_hooks = [
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=num_last_epochs,
        priority=48),
    dict(
        type='SyncRandomSizeHook',
        ratio_range=ratio_range,
        img_scale=img_scale,
        priority=48),
    dict(
        type='SyncNormHook',
        num_last_epochs=num_last_epochs,
        interval=interval,
        priority=48),
    dict(type='ExpMomentumEMAHook', resume_from=resume_from, priority=49)
]

checkpoint_config = dict(interval=interval)
evaluation = dict(metric=['bbox', 'track'], interval=interval)
search_metrics = ['MOTA', 'IDF1', 'FN', 'FP', 'IDs', 'MT', 'ML']

# you need to set mode='dynamic' if you are using pytorch<=1.5.0
fp16 = dict(loss_scale=dict(init_scale=512.))
