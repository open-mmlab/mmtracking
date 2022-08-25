_base_ = [
    '../../_base_/datasets/youtube_vis.py', '../../_base_/default_runtime.py'
]

num_classes = 40
num_frames = 2
model = dict(
    type='Mask2Former',
    data_preprocessor=dict(
        type='TrackDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=True,
        pad_size_divisor=32),
    backbone=dict(
        _scope_='mmdet',
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    track_head=dict(
        type='Mask2FormerHead',
        in_channels=[256, 512, 1024, 2048],  # pass to pixel_decoder inside
        strides=[4, 8, 16, 32],
        feat_channels=256,
        out_channels=256,
        num_classes=num_classes,
        num_queries=100,
        num_frames=num_frames,
        num_transformer_feat_level=3,
        pixel_decoder=dict(
            type='mmdet.MSDeformAttnPixelDecoder',
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(
                type='mmdet.DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        im2col_step=128,
                        dropout=0.0,
                        batch_first=False,
                        norm_cfg=None,
                        init_cfg=None),
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True)),
                    operation_order=('self_attn', 'norm', 'ffn', 'norm')),
                init_cfg=None),
            positional_encoding=dict(
                type='mmdet.SinePositionalEncoding',
                num_feats=128,
                normalize=True),
            init_cfg=None),
        enforce_decoder_input_project=False,
        positional_encoding=dict(
            type='SinePositionalEncoding3D', num_feats=128, normalize=True),
        transformer_decoder=dict(
            type='mmdet.DetrTransformerDecoder',
            return_intermediate=True,
            num_layers=9,
            transformerlayers=dict(
                type='mmdet.DetrTransformerDecoderLayer',
                attn_cfgs=dict(
                    type='MultiheadAttention',
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=False),
                ffn_cfgs=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True),
                feedforward_channels=2048,
                operation_order=('cross_attn', 'norm', 'self_attn', 'norm',
                                 'ffn', 'norm')),
            init_cfg=None),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * num_classes + [0.1]),
        loss_mask=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='mmdet.DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0),
        train_cfg=dict(
            num_points=12544,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            assigner=dict(
                type='mmdet.HungarianAssigner',
                match_costs=[
                    dict(type='mmdet.ClassificationCost', weight=2.0),
                    dict(
                        type='mmdet.CrossEntropyLossCost',
                        weight=5.0,
                        use_sigmoid=True),
                    dict(
                        type='mmdet.DiceCost',
                        weight=5.0,
                        pred_act=True,
                        eps=1.0)
                ]),
            sampler=dict(type='mmdet.MaskPseudoSampler'))),
    init_cfg=dict(
        type='Pretrained',
        checkpoint='https://download.openmmlab.com/mmdetection/v2.0/'
        'mask2former/mask2former_r50_lsj_8x2_50e_coco/'
        'mask2former_r50_lsj_8x2_50e_coco_20220506_191028-8e96e88b.pth'))

# optimizer
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
            'query_embed': embed_multi,
            'query_feat': embed_multi,
            'level_embed': embed_multi,
        },
        norm_decay_mult=0.0),
    clip_grad=dict(max_norm=0.01, norm_type=2))

# learning policy
max_iters = 6000
param_scheduler = dict(
    type='MultiStepLR',
    begin=0,
    end=max_iters,
    by_epoch=False,
    milestones=[
        4000,
    ],
    gamma=0.1)
# runtime settings
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=max_iters, val_interval=6001)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', by_epoch=False, save_last=True, interval=2000))
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=False)

train_pipeline = [
    dict(
        type='TransformBroadcaster',
        share_random_params=True,
        transforms=[
            dict(type='LoadImageFromFile'),
            dict(
                type='LoadTrackAnnotations',
                with_instance_id=True,
                with_mask=True,
                with_bbox=True),
            dict(type='mmdet.Resize', scale=(640, 360), keep_ratio=True),
            dict(type='mmdet.RandomFlip', prob=0.5),
        ]),
    dict(type='PackTrackInputs', num_key_frames=num_frames)
]
# dataloader
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        pipeline=train_pipeline,
        ref_img_sampler=dict(
            num_ref_imgs=1,
            frame_range=5,
            filter_key_img=True,
            method='uniform')))
val_dataloader = dict(
    num_workers=2,
    sampler=dict(type='VideoSampler'),
    batch_sampler=dict(type='EntireVideoBatchSampler'),
)
test_dataloader = val_dataloader

# evaluator
val_evaluator = dict(
    type='YouTubeVISMetric',
    metric='youtube_vis_ap',
    outfile_prefix='./youtube_vis_results',
    format_only=True)
test_evaluator = val_evaluator
