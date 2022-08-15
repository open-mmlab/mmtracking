# dataset settings
dataset_type = 'TaoDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadMultiImagesFromFile'),
    dict(type='SeqLoadAnnotations', with_bbox=True, with_track=True),
    dict(
        type='SeqResize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                   (1333, 768), (1333, 800)],
        share_params=True,
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
    dict(type='SeqNormalize', **img_norm_cfg),
    dict(type='SeqPad', size_divisor=32),
    dict(type='MatchInstances', skip_nomatch=True),
    dict(
        type='VideoCollect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_match_indices'],
    ),
    dict(type='SeqDefaultFormatBundle', ref_prefix='ref'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='VideoCollect', keys=['img'])
        ])
]
data_root = 'data/tao/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='ClassBalancedDataset',
        oversample_thr=1e-3,
        dataset=dict(
            type=dataset_type,
            classes=data_root + 'annotations/tao_classes.txt',
            load_as_video=False,
            ann_file='data/lvis/annotations/lvisv0.5+coco_train.json',
            img_prefix='data/lvis/train/',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        classes=data_root + 'annotations/tao_classes.txt',
        ann_file=data_root + 'annotations/validation_482_classes.json',
        img_prefix=data_root + 'val/',
        ref_img_sampler=None,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=data_root + 'annotations/tao_classes.txt',
        ann_file=data_root + 'annotations/validation_482_classes.json',
        img_prefix=data_root + 'val/',
        ref_img_sampler=None,
        pipeline=test_pipeline))
