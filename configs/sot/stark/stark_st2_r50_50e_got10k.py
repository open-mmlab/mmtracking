_base_ = ['./stark_st1_r50_500e_got10k.py']

# model setting
model = dict(
    type='Stark',
    head=dict(
        type='StarkHead',
        cls_head=dict(
            type='ScoreHead',
            input_dim=256,
            hidden_dim=256,
            output_dim=1,
            num_layers=3,
            use_bn=False),
        frozen_modules=['transformer', 'bbox_head', 'query_embedding'],
        loss_cls=dict(type='mmdet.CrossEntropyLoss', use_sigmoid=True)),
    frozen_modules=['backbone', 'neck'])

data_root = 'data/'
# the only difference of ``train_pipeline`` compared with that in stark_st1 is
# ``train_cls_head=True`` in ``TridentSampling``.
train_pipeline = [
    dict(
        type='TridentSampling',
        num_search_frames=1,
        num_template_frames=2,
        max_frame_range=[200],
        cls_pos_prob=0.5,
        train_cls_head=True),
    dict(
        type='TransformBroadcaster',
        share_random_params=True,
        transforms=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadTrackAnnotations', with_instance_id=False),
            dict(type='GrayAug', prob=0.05),
            dict(type='mmdet.RandomFlip', prob=0.5, direction='horizontal')
        ]),
    dict(
        type='SeqBboxJitter',
        center_jitter_factor=[0, 0, 4.5],
        scale_jitter_factor=[0, 0, 0.5],
        crop_size_factor=[2, 2, 5]),
    dict(
        type='SeqCropLikeStark',
        crop_size_factor=[2, 2, 5],
        output_size=[128, 128, 320]),
    dict(
        type='TransformBroadcaster',
        share_random_params=True,
        transforms=[dict(type='BrightnessAug', jitter_range=0.2)]),
    dict(
        type='TransformBroadcaster',
        share_random_params=False,
        transforms=[
            dict(type='mmdet.RandomFlip', prob=0.5, direction='horizontal')
        ]),
    dict(type='CheckPadMaskValidity', stride=16),
    dict(type='PackTrackInputs', ref_prefix='search', num_template_frames=2)
]

# dataset settings
train_dataloader = dict(
    dataset=dict(
        type='GOT10kDataset',
        data_root=data_root,
        ann_file='got10k/annotations/got10k_train_infos.txt',
        data_prefix=dict(img_path='got10k'),
        pipeline=train_pipeline,
        test_mode=False))

# runner loop
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=50, val_begin=50, val_interval=1)

# learning policy
param_scheduler = dict(type='MultiStepLR', milestones=[40], gamma=0.1)

# checkpoint saving
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=10))

load_from = 'logs/stark_st1_got10k_online/epoch_500.pth'
