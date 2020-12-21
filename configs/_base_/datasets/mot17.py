# dataset settings
dataset_type = 'MOT17Dataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadMultiImagesFromFile', to_float32=True),
    dict(type='SeqLoadAnnotations', with_bbox=True, with_track=True),
    dict(
        type='SeqResize',
        img_scale=(1088, 1088),
        share_params=True,
        ratio_range=(0.8, 1.2),
        keep_ratio=True,
        bbox_clip_border=False),
    dict(type='SeqPhotoMetricDistortion', share_params=True),
    dict(
        type='SeqRandomCrop',
        share_params=False,
        crop_size=(1088, 1088),
        bbox_clip_border=False),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
    dict(type='SeqNormalize', **img_norm_cfg),
    dict(type='SeqPad', size_divisor=32),
    dict(type='MatchInstances', skip_nomatch=True),
    dict(
        type='VideoCollect',
        keys=[
            'img', 'gt_bboxes', 'gt_labels', 'gt_match_indices',
            'gt_instance_ids'
        ]),
    dict(type='SeqDefaultFormatBundle', ref_prefix='ref')
]
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
data_root = 'data/MOT17/'
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        visibility_thr=0.25,
        ann_file=data_root + 'annotations/train_cocoformat.json',
        img_prefix=data_root + 'train',
        ref_img_sampler=dict(
            num_ref_imgs=1,
            frame_range=10,
            filter_key_img=True,
            method='uniform'),
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/train_cocoformat.json',
        img_prefix=data_root + 'train',
        ref_img_sampler=None,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/train_cocoformat.json',
        img_prefix=data_root + 'train',
        ref_img_sampler=None,
        detection_file=data_root + 'annotations/train_detections.pkl',
        pipeline=test_pipeline))
