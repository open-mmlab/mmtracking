# dataset settings
dataset_type = 'BDDVideoDataset'
data_root = 'data/bdd100k/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadMultiImagesFromFile'),
    dict(type='SeqLoadAnnotations', with_bbox=True, with_track=True),
    dict(
        type='SeqResize',
        share_params=True,
        ratio_range=(0.8, 1.2),
        img_scale=(1296, 720),
        keep_ratio=True),
    dict(type='SeqPhotoMetricDistortion', share_params=True),
    dict(type='SeqRandomCrop', share_params=False, crop_size=(600, 1080)),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
    dict(type='SeqNormalize', **img_norm_cfg),
    dict(type='SeqPad', size_divisor=32),
    dict(type='MatchInstances', skip_nomatch=True),
    dict(
        type='VideoCollect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_match_indices']),
    dict(type='SeqDefaultFormatBundle', ref_prefix='ref')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1296, 720),
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
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=[
        dict(
            type=dataset_type,
            load_as_video=True,
            ann_file=data_root +
            'tracking/annotations/bdd100k_track_train_cocoformat.json',
            img_prefix=data_root + 'tracking/train/',
            key_img_sampler=dict(interval=1),
            ref_img_sampler=dict(
                num_ref_imgs=1, frame_range=3, method='uniform'),
            pipeline=train_pipeline),
        dict(
            type=dataset_type,
            load_as_video=False,
            ann_file=data_root +
            'detection/annotations/bdd100k_det_train_cocoformat.json',
            img_prefix=data_root + 'detection/train/',
            pipeline=train_pipeline)
    ],
    val=dict(
        type=dataset_type,
        ann_file=data_root +
        'tracking/annotations/bdd100k_track_val_cocoformat.json',
        img_prefix=data_root + 'tracking/val/',
        ref_img_sampler=None,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root +
        'tracking/annotations/bdd100k_track_val_cocoformat.json',
        img_prefix=data_root + 'tracking/val/',
        ref_img_sampler=None,
        pipeline=test_pipeline))
