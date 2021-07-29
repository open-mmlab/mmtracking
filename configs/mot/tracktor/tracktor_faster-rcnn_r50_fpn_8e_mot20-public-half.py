_base_ = ['./tracktor_faster-rcnn_r50_fpn_4e_mot17-private-half.py']
model = dict(
    pretrains=dict(
        detector=  # noqa: E251
        'https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_8e_mot20-half-860a6c6f.pth',  # noqa: E501
        reid=  # noqa: E251
        'https://download.openmmlab.com/mmtracking/mot/reid/reid_r50_6e_mot20-afbdfea4.pth'  # noqa: E501
    ),
    detector=dict(
        rpn_head=dict(bbox_coder=dict(clip_border=True)),
        roi_head=dict(
            bbox_head=dict(bbox_coder=dict(clip_border=True), num_classes=1))),
    reid=dict(head=dict(num_classes=1705)))
data_root = 'data/MOT20/'
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
# learning policy
lr_config = dict(step=[6])
# runtime settings
total_epochs = 8
