# dataset settings
dataset_type = 'ImagenetVIDVideoDataset'
data_root = 'data/imagenet_vid/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadMultiImagesFromFile'),
    dict(type='SeqLoadAnnotations', with_bbox=True, with_track=True),
    dict(type='SeqResize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
    dict(type='SeqNormalize', **img_norm_cfg),
    dict(type='SeqPad', size_divisor=16),
    dict(type='ConcatVideoReferences'),
    dict(type='SeqDefaultFormatBundle'),
    dict(
        type='VideoCollect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_instance_ids'],
        ref_prefix='ref'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1000, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=16),
            dict(type='ConcatVideoReferences'),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='VideoCollect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    train=[
        dict(
            type=dataset_type,
            match_gts=False,
            ann_file=data_root + 'annotations/imagenet_vid_train.json',
            img_prefix=data_root + 'data/VID/',
            ref_img_sampler=dict(
                num_ref_imgs=2,
                frame_range=9,
                filter_key_frame=True,
                method='bilateral_uniform'),
            pipeline=train_pipeline),
        dict(
            type=dataset_type,
            match_gts=False,
            load_as_video=False,
            ann_file=data_root + 'annotations/imagenet_det_30cls.json',
            img_prefix=data_root + 'data/DET/',
            pipeline=train_pipeline)
    ],
    val=dict(
        type=dataset_type,
        match_gts=False,
        ann_file=data_root + 'annotations/imageNet_vid_val.json',
        img_prefix=data_root + 'data/VID/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        match_gts=False,
        ann_file=data_root + 'annotations/imageNet_vid_val.json',
        img_prefix=data_root + 'data/VID/',
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[3, 5])
# checkpoint saving
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 6
dist_params = dict(backend='nccl', port='12346')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
evaluation = dict(metric=['bbox'], interval=1)
