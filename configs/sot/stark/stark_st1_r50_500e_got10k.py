_base_ = ['../../_base_/default_runtime.py']

cudnn_benchmark = False
deterministic = True
seed = 1

# model settings
model = dict(
    type='Stark',
    data_preprocessor=dict(
        type='TrackDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=3,
        strides=(1, 2, 2),
        dilations=[1, 1, 1],
        out_indices=[2],
        frozen_stages=1,
        norm_eval=True,
        norm_cfg=dict(type='BN', requires_grad=False),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='mmdet.ChannelMapper',
        in_channels=[1024],
        out_channels=256,
        kernel_size=1,
        act_cfg=None),
    head=dict(
        type='StarkHead',
        num_querys=1,
        transformer=dict(
            type='StarkTransformer',
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            attn_drop=0.1,
                            dropout_layer=dict(type='Dropout', drop_prob=0.1))
                    ],
                    ffn_cfgs=dict(
                        feedforward_channels=2048,
                        embed_dims=256,
                        ffn_drop=0.1),
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DetrTransformerDecoder',
                return_intermediate=False,
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiheadAttention',
                        embed_dims=256,
                        num_heads=8,
                        attn_drop=0.1,
                        dropout_layer=dict(type='Dropout', drop_prob=0.1)),
                    ffn_cfgs=dict(
                        feedforward_channels=2048,
                        embed_dims=256,
                        ffn_drop=0.1),
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
        ),
        positional_encoding=dict(
            type='SinePositionalEncoding', num_feats=128, normalize=True),
        bbox_head=dict(
            type='CornerPredictorHead',
            inplanes=256,
            channel=256,
            feat_size=20,
            stride=16),
        loss_bbox=dict(type='mmdet.L1Loss', loss_weight=5.0),
        loss_iou=dict(type='mmdet.GIoULoss', loss_weight=2.0)),
    test_cfg=dict(
        search_factor=5.0,
        search_size=320,
        template_factor=2.0,
        template_size=128,
        update_intervals=[200]))

data_root = 'data/'
train_pipeline = [
    dict(
        type='TridentSampling',
        num_search_frames=1,
        num_template_frames=2,
        max_frame_range=[200],
        cls_pos_prob=0.5,
        train_cls_head=False),
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

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadTrackAnnotations', with_instance_id=False),
    dict(type='PackTrackInputs', pack_single_img=True)
]

# dataset settings
train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='QuotaSampler', samples_per_epoch=60000),
    dataset=dict(
        type='GOT10kDataset',
        data_root=data_root,
        ann_file='got10k/annotations/got10k_train_infos.txt',
        data_prefix=dict(img_path='got10k'),
        pipeline=train_pipeline,
        test_mode=False))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='VideoSampler'),
    dataset=dict(
        type='GOT10kDataset',
        data_root=data_root,
        ann_file='got10k/annotations/got10k_test_infos.txt',
        data_prefix=dict(img_path='got10k'),
        pipeline=test_pipeline,
        test_mode=True))
test_dataloader = val_dataloader

# evaluator
val_evaluator = dict(
    type='SOTMetric',
    metric='OPE',
    format_only=True,
    metric_options=dict(dataset_type='got10k'))
test_evaluator = val_evaluator

# runner loop
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=500, val_begin=500, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning policy
param_scheduler = dict(type='MultiStepLR', milestones=[400], gamma=0.1)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys=dict(backbone=dict(lr_mult=0.1, decay_mult=1.0))))

# checkpoint saving
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=100))
