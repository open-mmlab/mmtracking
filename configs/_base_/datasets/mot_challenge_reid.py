dataset_type = 'ReIDDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadMultiImagesFromFile', to_float32=True),
    dict(
        type='SeqResize',
        img_scale=(128, 256),
        share_params=False,
        keep_ratio=False,
        bbox_clip_border=False,
        override=False),
    dict(
        type='SeqRandomFlip',
        share_params=False,
        flip_ratio=0.5,
        direction='horizontal'),
    dict(type='SeqNormalize', **img_norm_cfg),
    dict(type='VideoCollect', keys=['img', 'gt_label']),
    dict(type='ReIDFormatBundle')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', img_scale=(128, 256), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'], meta_keys=[])
]
data_root = 'data/MOT17/'
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        triplet_sampler=dict(num_ids=8, ins_per_id=4),
        data_prefix=data_root + 'reid/imgs',
        ann_file=data_root + 'reid/meta/train_80.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        triplet_sampler=None,
        data_prefix=data_root + 'reid/imgs',
        ann_file=data_root + 'reid/meta/val_20.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        triplet_sampler=None,
        data_prefix=data_root + 'reid/imgs',
        ann_file=data_root + 'reid/meta/val_20.txt',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='mAP')
