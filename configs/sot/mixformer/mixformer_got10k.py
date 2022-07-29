
cudnn_benchmark = False
deterministic = True
seed = 1

# model setting
model = dict(
    type='MixFormer',
    backbone=dict(
        type='ConvolutionalVisionTransformer',
        spec=dict(
            NUM_STAGES=3,
            PATCH_SIZE=[7, 3, 3],
            PATCH_STRIDE=[4, 2, 2],
            PATCH_PADDING=[2, 1, 1],
            DIM_EMBED=[64, 192, 384],
            NUM_HEADS=[1, 3, 6],
            DEPTH=[1, 4, 16],
            MLP_RATIO=[4, 4, 4],
            ATTN_DROP_RATE=[0.0, 0.0, 0.0],
            DROP_RATE=[0.0, 0.0, 0.0],
            DROP_PATH_RATE=[0.0, 0.0, 0.1],
            QKV_BIAS=[True, True, True],
            CLS_TOKEN=[False, False, False],
            POS_EMBED=[False, False, False],
            QKV_PROJ_METHOD=['dw_bn', 'dw_bn', 'dw_bn'],
            KERNEL_QKV=[3, 3, 3],
            PADDING_KV=[1, 1, 1],
            STRIDE_KV=[2, 2, 2],
            PADDING_Q=[1, 1, 1],
            STRIDE_Q=[1, 1, 1],
            FREEZE_BN=True
        )
    ),
    head=dict(
        type='MixFormerHead',
        bbox_head=dict(
            type='MixformerCornerPredictorHead',
            inplanes=384,
            channel=384,
            feat_sz=20,
            stride=16,
            freeze_bn=False
        ),
        score_head=dict(
            type='ScoreDecoder',
            pool_size=4,
            num_heads=6,
            hidden_dim=384,
            num_layers=3
        )
   ),
    test_cfg=dict(
        search_factor=4.5,
        search_size=320,
        template_factor=2.0,
        template_size=128,
        update_interval=[200],
        online_size=[2],
    )
)

data_root = 'data/'
train_pipeline = [
    dict(
        type='TridentSampling',
        num_search_frames=1,
        num_template_frames=2,
        max_frame_range=[200],
        cls_pos_prob=0.5,
        train_cls_head=False),
    dict(type='LoadMultiImagesFromFile', to_float32=True),
    dict(type='SeqLoadAnnotations', with_bbox=True, with_label=False),
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
    dict(type='SeqBrightnessAug', jitter_range=0.2),
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
    dict(type='CheckPadMaskValidity', stride=16),
    dict(
        type='VideoCollect',
        keys=['img', 'gt_bboxes', 'padding_mask'],
        meta_keys=('valid')),
    dict(type='ConcatSameTypeFrames', num_key_frames=2),
    dict(type='SeqDefaultFormatBundle', ref_prefix='search')
]

img_norm_cfg = dict(mean=[0, 0, 0], std=[1, 1, 1], to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_label=False),
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
    workers_per_gpu=8,
    persistent_workers=True,
    samples_per_epoch=60000,
    train=dict(
        type='RandomSampleConcatDataset',
        dataset_sampling_weights=[1],
        dataset_cfgs=[
            dict(
                type='GOT10kDataset',
                ann_file=data_root +
                'got10k/annotations/got10k_train_infos.txt',
                img_prefix=data_root + 'got10k',
                pipeline=train_pipeline,
                split='train',
                test_mode=False)
        ]),
    val=dict(
        type='GOT10kDataset',
        ann_file=data_root + 'got10k/annotations/got10k_test_infos.txt',
        img_prefix=data_root + 'got10k',
        pipeline=test_pipeline,
        split='test',
        test_mode=True),
    # test=dict(
    #     type='GOT10kDataset',
    #     ann_file=data_root + 'got10k/annotations/got10k_test_infos.txt',
    #     img_prefix=data_root + 'got10k',
    #     pipeline=test_pipeline,
    #     split='test',
    #     test_mode=True))
    test=dict(
        type='LaSOTDataset',
        ann_file=data_root + 'lasot/annotations/lasot_test_infos.txt',
        img_prefix=data_root + 'lasot/LaSOTBenchmark',
        pipeline=test_pipeline,
        split='test',
        test_mode=True,
        only_eval_visible=True))

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
checkpoint_config = dict(interval=100)
evaluation = dict(
    metric=['track'],
    interval=100,
    start=501,
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
