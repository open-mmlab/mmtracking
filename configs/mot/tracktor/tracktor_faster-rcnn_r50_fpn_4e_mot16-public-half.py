_base_ = ['./tracktor_faster-rcnn_r50_fpn_4e_mot17-private-half.py']

model = dict(
    detector=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot16-half_20210804_001054-73477869.pth'  # noqa: E501
        )),
    reid=dict(
        head=dict(num_classes=375),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://download.openmmlab.com/mmtracking/mot/reid/reid_r50_6e_mot16_20210803_204826-1b3e3cfd.pth'  # noqa: E501
        )))
# data
data_root = 'data/MOT16/'
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
    train=dict(
        ann_file=data_root + 'annotations/half-train_cocoformat.json',
        detection_file=data_root + 'annotations/half-train_detections.pkl',
        img_prefix=data_root + 'train'),
    val=dict(
        ann_file=data_root + 'annotations/half-val_cocoformat.json',
        detection_file=data_root + 'annotations/half-val_detections.pkl',
        img_prefix=data_root + 'train',
        pipeline=test_pipeline),
    test=dict(
        ann_file=data_root + 'annotations/half-val_cocoformat.json',
        detection_file=data_root + 'annotations/half-val_detections.pkl',
        img_prefix=data_root + 'train',
        pipeline=test_pipeline))
