_base_ = ['./ocsort_yolox_x_crowdhuman_mot17-private-half.py']

img_scale = (896, 1600)

model = dict(
    detector=dict(input_size=img_scale, random_size_range=(20, 36)),
    tracker=dict(
        weight_iou_with_det_scores=False,
        match_iou_thr=0.3,
    ))

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
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(type='Pad', size_divisor=32, pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
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
            dict(
                type='Normalize',
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(
                type='Pad',
                size_divisor=32,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='VideoCollect', keys=['img'])
        ])
]
data = dict(
    train=dict(
        dataset=dict(
            ann_file=[
                'data/MOT20/annotations/train_cocoformat.json',
                'data/crowdhuman/annotations/crowdhuman_train.json',
                'data/crowdhuman/annotations/crowdhuman_val.json'
            ],
            img_prefix=[
                'data/MOT20/train', 'data/crowdhuman/train',
                'data/crowdhuman/val'
            ]),
        pipeline=train_pipeline),
    val=dict(
        ann_file='data/MOT17/annotations/train_cocoformat.json',
        img_prefix='data/MOT17/train',
        pipeline=test_pipeline),
    test=dict(
        ann_file='data/MOT20/annotations/test_cocoformat.json',
        img_prefix='data/MOT20/test',
        pipeline=test_pipeline))

checkpoint_config = dict(interval=1)
evaluation = dict(metric=['bbox', 'track'], interval=1)
