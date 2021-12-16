cudnn_benchmark = True
persistent_workers = True
crop_size = 511
exemplar_size = 127
search_size = 255

# model setting
model = dict(
    type='Stark',
    backbone=dict(
        type='ResNet',
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
        type='ChannelMapper',
        in_channels=[1024],
        out_channels=256,
        kernel_size=1,
        act_cfg=None),
    head=dict(
        type='StarkHead',
        num_querys=1,
        run_bbox_head=True,
        run_cls_head=False,
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
        cls_head=dict(
            type='ScoreHead',
            input_dim=256,
            hidden_dim=256,
            output_dim=1,
            num_layers=3,
            BN=False),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    test_cfg=dict(
        search_factor=5.0,
        search_size=320,
        template_factor=2.0,
        template_size=128,
        update_intervals=[200]))

data_root = 'data/'
train_pipeline = [
    dict(type='LoadMultiImagesFromFile', to_float32=True),
    dict(type='SeqLoadAnnotations', with_bbox=True),
    dict(type='SeqGrayAug', prob=0.05),
    dict(
        type='SeqRandomFlip',
        share_params=True,
        flip_ratio=0.5,
        direction='horizontal'),
    dict(
        type='SeqBboxJitter',
        center_jitter_factor=[0, 0, 4.5],
        scale_jitter_factor=[0, 0, 0.5],
        crop_size_factor=[2, 2, 5]),
    dict(
        type='SeqCropLikeStark',
        crop_size_factor=[2, 2, 5],
        output_size=[128, 128, 320]),
    dict(type='SeqBrightnessAug', brightness_jitter=0.2),
    dict(
        type='SeqRandomFlip',
        share_params=False,
        flip_ratio=0.5,
        direction='horizontal'),
    dict(
        type='SeqNormalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='VideoCollect', keys=['img', 'gt_bboxes', 'att_mask']),
    dict(type='ConcatVideoTripleReferences'),
    dict(type='SeqDefaultFormatBundle', ref_prefix='search')
]

img_norm_cfg = dict(mean=[0, 0, 0], std=[1, 1, 1], to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='MultiScaleFlipAug',
        scale_factor=1,
        flip=False,
        transforms=[
            dict(type='Normalize', **img_norm_cfg),
            dict(type='VideoCollect', keys=['img', 'gt_bboxes']),
            dict(type='ImageToTensor', keys=['img'])
        ])
]

# dataset settings
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    persistent_workers=False,
    train=[
        dict(datasets_sampling_prob=[1], train_cls=False),
        dict(
            type='SOTQuotaTrainDataset',
            ann_file=data_root + 'got10k/annotations/got10k_train.json',
            img_prefix=data_root + 'got10k/train',
            pipeline=train_pipeline,
            max_gap=[200],
            num_search_frames=1,
            num_template_frames=2,
            visible_keys=['absence', 'cover'],
            ref_img_sampler=None,
            test_mode=False),
        # dict(
        #     type='SOTQuotaTrainDataset',
        #     ann_file=data_root + 'lasot/annotations/lasot_train.json',
        #     img_prefix=data_root + 'lasot/LaSOTBenchmark',
        #     pipeline=train_pipeline,
        #     max_gap=[200],
        #     num_search_frames=1,
        #     num_template_frames=2,
        #     visible_keys=['full_occlusion', 'out_of_view'],
        #     ref_img_sampler=None,
        #     test_mode=False),
        # dict(
        #     type='SOTQuotaTrainDataset',
        #     ann_file=data_root +
        #     'trackingnet/annotations/trackingnet_train.json',
        #     img_prefix=data_root + 'trackingnet/train',
        #     pipeline=train_pipeline,
        #     max_gap=[200],
        #     num_search_frames=1,
        #     num_template_frames=2,
        #     visible_keys=None,
        #     ref_img_sampler=None,
        #     test_mode=False),
        # dict(
        #     type='SOTQuotaTrainDataset',
        #     ann_file=data_root + 'coco/annotations/instances_train2017.json',
        #     img_prefix=data_root + 'coco/train2017',
        #     pipeline=train_pipeline,
        #     max_gap=[200],
        #     num_search_frames=1,
        #     num_template_frames=2,
        #     visible_keys=None,
        #     ref_img_sampler=None,
        #     test_mode=False),
    ],
    val=dict(
        type='GOT10kDataset',
        test_load_ann=True,
        ann_file=data_root + 'got10k/annotations/got10k_val.json',
        img_prefix=data_root + 'got10k/val',
        pipeline=test_pipeline,
        ref_img_sampler=None,
        test_mode=True),
    test=dict(
        type='GOT10kDataset',
        test_load_ann=True,
        ann_file=data_root + 'got10k/annotations/got10k_test.json',
        img_prefix=data_root + 'got10k/test',
        pipeline=test_pipeline,
        ref_img_sampler=None,
        test_mode=True))

# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys=dict(backbone=dict(lr_mult=0.1, decay_mult=1.0))))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[400])
# checkpoint saving
checkpoint_config = dict(interval=25)
evaluation = dict(
    metric=['track'],
    interval=100,
    start=500,
    rule='greater',
    save_best='success')
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 500
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/xxx'
load_from = None
resume_from = None
workflow = [('train', 1)]
