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
        num_classes=1,
        in_channels=256,
        stride=[16],
        num_cls_fcs=3,
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
                    feedforward_channels=2048,
                    ffn_cfgs=dict(embed_dims=256, ffn_drop=0.1),
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DetrTransformerDecoder',
                return_intermediate=False,
                num_layers=6,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=dict(
                        type='MultiheadAttention',
                        embed_dims=256,
                        num_heads=8,
                        attn_drop=0.1,
                        dropout_layer=dict(type='Dropout', drop_prob=0.1)),
                    feedforward_channels=2048,
                    ffn_cfgs=dict(embed_dims=256, ffn_drop=0.1),
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
        ),
        positional_encoding=dict(
            type='SinePositionalEncoding', num_feats=128, normalize=True)),
    test_cfg=dict(
        epoch=50,
        search_factor=5.0,
        search_size=320,
        template_factor=2.0,
        template_size=128,
        update_intervals=[25]))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='MultiScaleFlipAug',
        scale_factor=1,
        flip=False,
        transforms=[
            dict(type='VideoCollect', keys=['img', 'gt_bboxes']),
            dict(type='ImageToTensor', keys=['img'])
        ])
]

data_root = 'data/'
# dataset settings
data = dict(
    samples_per_gpu=28,
    workers_per_gpu=2,
    test=dict(
        type='TrackingNetDataset',
        test_load_ann=True,
        ann_file=data_root +
        'trackingnet/TEST/annotations/trackingnet_test.json',
        img_prefix=data_root + 'trackingnet/TEST/frames',
        pipeline=test_pipeline,
        ref_img_sampler=None,
        test_mode=True))
