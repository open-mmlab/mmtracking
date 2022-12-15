_base_ = [
    '../../_base_/models/yolox_x_8x8.py',
    '../../_base_/datasets/mot_challenge.py', '../../_base_/default_runtime.py'
]

dataset_type = 'MOTChallengeDataset'
data_root = 'data/MOT17/'

img_scale = (800, 1440)
batch_size = 4

model = dict(
    type='ByteTrack',
    data_preprocessor=dict(
        type='TrackDataPreprocessor',
        pad_size_divisor=32,
        batch_augments=[
            dict(
                type='mmdet.BatchSyncRandomResize',
                random_size_range=(576, 1024),
                size_divisor=32,
                interval=10)
        ]),
    detector=dict(
        _scope_='mmdet',
        bbox_head=dict(num_classes=1),
        test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.7)),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_x_8x8_300e_coco/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth'  # noqa: E501
        )),
    motion=dict(type='KalmanFilter'),
    tracker=dict(
        type='ByteTracker',
        obj_score_thrs=dict(high=0.6, low=0.1),
        init_track_thr=0.7,
        weight_iou_with_det_scores=True,
        match_iou_thrs=dict(high=0.1, low=0.5, tentative=0.3),
        num_frames_retain=30))

train_pipeline = [
    dict(
        type='mmdet.Mosaic',
        img_scale=img_scale,
        pad_val=114.0,
        bbox_clip_border=False),
    dict(
        type='mmdet.RandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        bbox_clip_border=False),
    dict(
        type='mmdet.MixUp',
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0,
        bbox_clip_border=False),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.Resize',
        scale=img_scale,
        keep_ratio=True,
        clip_object_border=False),
    dict(
        type='mmdet.Pad',
        size_divisor=32,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(
        type='mmdet.FilterAnnotations',
        min_gt_bbox_wh=(1, 1),
        keep_empty=False),
    dict(type='PackTrackInputs', pack_single_img=True)
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='mmdet.Pad',
        size_divisor=32,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='PackTrackInputs', pack_single_img=True)
]

train_dataloader = dict(
    _delete_=True,
    batch_size=batch_size,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='mmdet.MultiImageMixDataset',
        dataset=dict(
            type='mmdet.ConcatDataset',
            datasets=[
                dict(
                    type='mmdet.CocoDataset',
                    data_root='data/MOT17',
                    ann_file='annotations/half-train_cocoformat.json',
                    # TODO: mmdet use img as key, but img_path is needed
                    data_prefix=dict(img='train'),
                    filter_cfg=dict(filter_empty_gt=True, min_size=32),
                    metainfo=dict(classes=('pedestrian')),
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadTrackAnnotations'),
                    ]),
                dict(
                    type='mmdet.CocoDataset',
                    data_root='data/crowdhuman',
                    ann_file='annotations/crowdhuman_train.json',
                    data_prefix=dict(img='train'),
                    filter_cfg=dict(filter_empty_gt=True, min_size=32),
                    metainfo=dict(classes=('pedestrian')),
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadTrackAnnotations'),
                    ]),
                dict(
                    type='mmdet.CocoDataset',
                    data_root='data/crowdhuman',
                    ann_file='annotations/crowdhuman_val.json',
                    data_prefix=dict(img='val'),
                    filter_cfg=dict(filter_empty_gt=True, min_size=32),
                    metainfo=dict(classes=('pedestrian')),
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadTrackAnnotations'),
                    ]),
            ]),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='VideoSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/half-val_cocoformat.json',
        data_prefix=dict(img_path='train'),
        ref_img_sampler=None,
        load_as_video=True,
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader
# optimizer
# default 8 gpu
lr = 0.001 / 8 * batch_size
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD', lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True),
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))

# some hyper parameters
# training settings
total_epochs = 80
num_last_epochs = 10
resume_from = None
interval = 5

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=total_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
# learning policy
param_scheduler = [
    dict(
        # use quadratic formula to warm up 1 epochs
        # and lr is updated by iteration
        # TODO: fix default scope in get function
        type='mmdet.QuadraticWarmupLR',
        by_epoch=True,
        begin=0,
        end=1,
        convert_to_iter_based=True),
    dict(
        # use cosine lr from 1 to 70 epoch
        type='mmdet.CosineAnnealingLR',
        eta_min=lr * 0.05,
        begin=1,
        T_max=total_epochs - num_last_epochs,
        end=total_epochs - num_last_epochs,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        # use fixed lr during last 10 epochs
        type='mmdet.ConstantLR',
        by_epoch=True,
        factor=1,
        begin=total_epochs - num_last_epochs,
        end=total_epochs,
    )
]

custom_hooks = [
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=num_last_epochs,
        priority=48),
    dict(type='mmdet.SyncNormHook', priority=48),
    dict(
        type='mmdet.EMAHook',
        ema_type='mmdet.ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        priority=49)
]
default_hooks = dict(checkpoint=dict(interval=1))
# evaluator
val_evaluator = dict(postprocess_tracklet_cfg=[
    dict(type='InterpolateTracklets', min_num_frames=5, max_num_frames=20)
])
test_evaluator = val_evaluator
