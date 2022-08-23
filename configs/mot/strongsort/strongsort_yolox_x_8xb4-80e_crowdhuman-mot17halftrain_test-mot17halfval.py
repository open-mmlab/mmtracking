_base_ = [
    '../../_base_/models/yolox_x_8x8.py',
    '../../_base_/datasets/mot_challenge.py', '../../_base_/default_runtime.py'
]

model = dict(
    type='StrongSORT',
    data_preprocessor=dict(
        type='TrackDataPreprocessor',
        pad_size_divisor=32,
        batch_augments=[
            dict(
                type='mmdet.BatchSyncRandomResize',
                random_size_range=(480, 800),
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
            'https://download.openmmlab.com/mmtracking/mot/strongsort/mot_dataset/yolox_x_crowdhuman_mot17-private-half_20220812_192036-b6c9ce9a.pth'  # noqa: E501
        )),
    kalman=dict(type='KalmanFilter', center_only=False, use_nsa=True),
    cmc=dict(
        type='CameraMotionCompensation',
        warp_mode='cv2.MOTION_EUCLIDEAN',
        num_iters=100,
        stop_eps=0.00001),
    reid=dict(
        type='BaseReID',
        data_preprocessor=None,
        backbone=dict(
            type='mmcls.ResNet',
            depth=50,
            num_stages=4,
            out_indices=(3, ),
            style='pytorch'),
        neck=dict(type='GlobalAveragePooling', kernel_size=(8, 4), stride=1),
        head=dict(
            type='LinearReIDHead',
            num_fcs=1,
            in_channels=2048,
            fc_channels=1024,
            out_channels=128,
            num_classes=380,
            loss_cls=dict(type='mmcls.CrossEntropyLoss', loss_weight=1.0),
            loss_triplet=dict(type='TripletLoss', margin=0.3, loss_weight=1.0),
            norm_cfg=dict(type='BN1d'),
            act_cfg=dict(type='ReLU')),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://download.openmmlab.com/mmtracking/mot/reid/reid_r50_6e_mot17-4bf6b63d.pth'  # noqa: E501
        )),
    tracker=dict(
        type='StrongSORTTracker',
        obj_score_thr=0.6,
        reid=dict(
            num_samples=None,
            img_scale=(256, 128),
            img_norm_cfg=dict(
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            match_score_thr=0.3,
            motion_weight=0.02,
        ),
        match_iou_thr=0.7,
        momentums=dict(embeds=0.1, ),
        num_tentatives=2,
        num_frames_retain=100))

dataset_type = 'MOTChallengeDataset'
data_root = 'data/MOT17/'
img_scale = (800, 1440)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='mmdet.Pad',
        size_divisor=32,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='PackTrackInputs', pack_single_img=True)
]

train_dataloader = None
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

train_cfg = None
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# evaluator
val_evaluator = dict(postprocess_tracklet_cfg=[
    dict(
        type='AppearanceFreeLink',
        checkpoint=  # noqa: E251
        'https://download.openmmlab.com/mmtracking/mot/strongsort/mot_dataset/aflink_motchallenge_20220812_190310-a7578ad3.pth',  # noqa: E501
        temporal_threshold=(0, 30),
        spatial_threshold=50,
        confidence_threshold=0.95,
    ),
    dict(
        type='InterpolateTracklets',
        min_num_frames=5,
        max_num_frames=20,
        use_gsi=True,
        smooth_tau=10)
])
test_evaluator = val_evaluator
