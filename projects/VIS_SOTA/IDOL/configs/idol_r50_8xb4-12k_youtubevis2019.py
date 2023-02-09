_base_ = [
    '../../../../configs/_base_/datasets/youtube_vis.py',  # noqa: E501
    '../../../../configs/_base_/default_runtime.py'
]

custom_imports = dict(imports=['projects.VIS_SOTA.IDOL.idol_src'], )

model = dict(
    type='IDOL',
    data_preprocessor=dict(
        type='TrackDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=True,
        pad_size_divisor=32),
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='mmdet.ChannelMapper',
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    track_head=dict(
        _scope_='mmdet',
        type='mmtrack.IDOLTrackHead',
        num_query=300,
        num_classes=40,
        in_channels=2048,
        with_box_refine=True,
        sync_cls_avg_factor=True,
        as_two_stage=False,
        transformer=dict(
            type='mmtrack.DeformableDetrTransformer',
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention', embed_dims=256),
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DeformableDetrTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=256)
                    ],
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0),
        loss_mask=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            reduction='mean',
            loss_weight=20.0),
        loss_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=1.0),
        loss_track=dict(
            type='mmtrack.MultiPosCrossEntropyLoss', loss_weight=0.25),
        loss_track_aux=dict(
            type='mmtrack.L2Loss',
            neg_pos_ub=3,
            pos_margin=0,
            neg_margin=0.1,
            hard_mining=True,
            loss_weight=1.0)),
    tracker=dict(
        type='IDOLTracker',
        init_score_thr=0.2,
        obj_score_thr=0.1,
        nms_thr_pre=0.5,
        nms_thr_post=0.05,
        addnew_score_thr=0.2,
        memo_tracklet_frames=10,
        memo_momentum=0.8,
        long_match=True,
        frame_weight=True,
        temporal_weight=True,
        memory_len=3,
        match_metric='bisoftmax'),
    # training and testing settings
    # can't del 'mmtrack'
    train_cfg=dict(
        assigner=dict(
            type='mmtrack.SimOTAAssigner',
            center_radius=2.5,
            match_costs=[
                dict(type='FocalLossCost', weight=1.0),
                dict(type='IoUCost', iou_mode='giou', weight=3.0)
            ]),
        cur_train_mode='VIS'),
)

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
    dict(type='PackTrackInputs', num_key_frames=2)
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadTrackAnnotations',
        with_instance_id=True,
        with_mask=True,
        with_bbox=True),
    # dict(type='mmdet.Resize', scale=(480, 1000), keep_ratio=True),
    dict(type='PackTrackInputs', pack_single_img=True)
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
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader
# evaluator
val_evaluator = dict(
    type='YouTubeVISMetric',
    metric='youtube_vis_ap',
    outfile_prefix='./youtube_vis_results',
    format_only=True)
test_evaluator = val_evaluator
