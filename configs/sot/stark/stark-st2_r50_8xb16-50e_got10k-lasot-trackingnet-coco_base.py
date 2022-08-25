_base_ = ['./stark-st1_r50_8xb16-500e_got10k-lasot-trackingnet-coco_base.py']

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

data_root = {{_base_.data_root}}
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
    dict(type='CheckPadMaskValidity', stride=16),
    dict(type='PackTrackInputs', ref_prefix='search', num_template_frames=2)
]

# dataset settings
train_dataloader = dict(
    dataset=dict(datasets=[
        dict(
            type='GOT10kDataset',
            data_root=data_root,
            ann_file='GOT10k/annotations/got10k_train_vot_infos.txt',
            data_prefix=dict(img_path='GOT10k'),
            pipeline=train_pipeline,
            test_mode=False),
        dict(
            type='LaSOTDataset',
            data_root=data_root,
            ann_file='LaSOT_full/annotations/lasot_train_infos.txt',
            data_prefix=dict(img_path='LaSOT_full/LaSOTBenchmark'),
            pipeline=train_pipeline,
            test_mode=False),
        dict(
            type='TrackingNetDataset',
            data_root=data_root,
            ann_file='TrackingNet/annotations/trackingnet_train_infos.txt',
            data_prefix=dict(img_path='TrackingNet'),
            pipeline=train_pipeline,
            test_mode=False),
        dict(
            type='SOTCocoDataset',
            data_root=data_root,
            ann_file='coco/annotations/instances_train2017.json',
            data_prefix=dict(img_path='coco/train2017'),
            pipeline=train_pipeline,
            test_mode=False)
    ]))

# runner loop
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=50, val_begin=50, val_interval=1)

# learning policy
param_scheduler = dict(type='MultiStepLR', milestones=[40], gamma=0.1)

# checkpoint saving
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=10))
