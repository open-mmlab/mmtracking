dataset_type = 'ReIDDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadMultiImagesFromFile', to_float32=True),
    dict(type='SeqResize', img_scale=(256, 128), share_params=False, keep_ratio=False),
    dict(type='SeqRandomFlip', share_params=False, flip_ratio=0.5, direction='horizontal'),
    dict(type='SeqNormalize', **img_norm_cfg),
    dict(type='SeqImageToTensor', keys=['img']),
    # dict(type='MultiImagesToTensor'),
    dict(type='SeqToTensor', keys=['gt_label']),
    dict(type='VideoCollect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, 128)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        triplet_sampler=dict(num_ids=8, ins_per_id=4),
        data_prefix='data/MOT17/reid/img',
        ann_file='data/MOT17/reid/meta/train.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        triplet_sampler=None,
        data_prefix='data/MOT17/reid/img',
        ann_file='data/MOT17/reid/meta/val.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        triplet_sampler=None,
        data_prefix='data/MOT17/reid/img',
        ann_file='data/MOT17/reid/meta/val.txt',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='mAP')
