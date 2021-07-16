_base_ = [
    '../../_base_/models/faster_rcnn_r50_fpn.py',
    '../../_base_/datasets/mot_challenge.py', '../../_base_/default_runtime.py'
]
model = dict(
    type='Tracktor',
    pretrains=dict(
        detector=  # noqa: E251
        'work_dirs/publish_model/faster-rcnn_r50_fpn_8e_mot20-half-860a6c6f.pth',  # noqa: E501
        reid=  # noqa: E251
        'work_dirs/resnet50_b32*8_MOT20_07081857/latest.pth'  # noqa: E501
    ),
    detector=dict(
        rpn_head=dict(bbox_coder=dict(clip_border=True)),
        roi_head=dict(
            bbox_head=dict(bbox_coder=dict(clip_border=True), num_classes=1))),
    reid=dict(
        type='BaseReID',
        backbone=dict(
            type='ResNet',
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
            num_classes=1704,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
            loss_pairwise=dict(
                type='TripletLoss', margin=0.3, loss_weight=1.0),
            norm_cfg=dict(type='BN1d'),
            act_cfg=dict(type='ReLU'))),
    motion=dict(
        type='CameraMotionCompensation',
        warp_mode='cv2.MOTION_EUCLIDEAN',
        num_iters=100,
        stop_eps=0.00001),
    tracker=dict(
        type='TracktorTracker',
        obj_score_thr=0.5,
        regression=dict(
            obj_score_thr=0.5,
            nms=dict(type='nms', iou_threshold=0.6),
            match_iou_thr=0.3),
        reid=dict(
            num_samples=10,
            img_scale=(256, 128),
            img_norm_cfg=None,
            match_score_thr=2.0,
            match_iou_thr=0.2),
        momentums=None,
        num_frames_retain=10))
# data
data_root = 'data/MOT20/'
data = dict(
    train=dict(
        ann_file=data_root + 'annotations/half-train_cocoformat.json',
        img_prefix=data_root + 'train'),
    val=dict(
        ann_file=data_root + 'annotations/half-val_cocoformat.json',
        img_prefix=data_root + 'train'),
    test=dict(
        ann_file=data_root + 'annotations/half-val_cocoformat.json',
        img_prefix=data_root + 'train'))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=1.0 / 100,
    step=[3])
# runtime settings
total_epochs = 4
evaluation = dict(metric=['bbox', 'track'], interval=1)
search_metrics = ['MOTA', 'IDF1', 'FN', 'FP', 'IDs', 'MT', 'ML']
