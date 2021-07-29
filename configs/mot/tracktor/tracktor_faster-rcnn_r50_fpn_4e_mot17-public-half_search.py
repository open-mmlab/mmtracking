_base_ = ['./tracktor_faster-rcnn_r50_fpn_4e_mot17-private-half.py']
model = dict(
    tracker=dict(
        type='TracktorTracker',
        obj_score_thr=[0.4, 0.5, 0.6],
        regression=dict(
            obj_score_thr=[0.4, 0.5, 0.6],
            nms=dict(type='nms', iou_threshold=0.6),
            match_iou_thr=[0.3, 0.5]),
        reid=dict(
            num_samples=10,
            img_scale=(256, 128),
            img_norm_cfg=None,
            match_score_thr=2.0,
            match_iou_thr=0.2),
        momentums=None,
        num_frames_retain=10))
# data
data_root = 'data/MOT17/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadDetections'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1088, 1088),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='VideoCollect', keys=['img', 'public_bboxes'])
        ])
]
data = dict(
    val=dict(
        detection_file=data_root + 'annotations/half-val_detections.pkl',
        pipeline=test_pipeline),
    test=dict(
        detection_file=data_root + 'annotations/half-val_detections.pkl',
        pipeline=test_pipeline))
