# SKIP REVIEW
# find_unused_parameters = True
# cudnn_benchmark = True
# model settings
model = dict(
    type='SiamRPN',
    pretrains=dict(backbone='/mnt/lustre/gongtao/Tracking_Code/MMTRACK/'
                   'pretrained_models/resnet50.model'),
    backbone=dict(
        type='SOTResNet',
        depth=50,
        out_indices=(1, 2, 3),
        frozen_stages=4,
        strides=(1, 2, 1, 1),
        dilations=(1, 1, 2, 4),
        norm_eval=True),
    neck=dict(
        type='ChannelMapper',
        in_channels=[512, 1024, 2048],
        out_channels=256,
        kernel_size=1,
        norm_cfg=dict(type='BN'),
        act_cfg=None),
    head=dict(
        type='SiameseRPNHead',
        anchor_generator=dict(
            type='SiameseRPNAnchorGenerator',
            strides=[8],
            ratios=[0.33, 0.5, 1, 2, 3],
            scales=[8]),
        in_channels=[256, 256, 256],
        weighted_sum=True))
train_cfg = dict(cls_weight=1.0, loc_weight=1.2)
test_cfg = dict(
    exemplar_size=127,
    instance_size=255,
    context_amount=0.5,
    center_size=7,
    rpn=dict(penalty_k=0.05, window_influence=0.42, lr=0.38))

data_root = 'data/sot/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(
        type='VideoCollect',
        keys=['img', 'gt_bboxes'],
        meta_keys=('is_video_data', 'frame_id'),
        default_meta_keys=('filename', 'ori_filename', 'ori_shape',
                           'img_shape'))
]
# dataset settings
data = dict(
    samples_per_gpu=28,
    workers_per_gpu=2,
    train=dict(
        type='LaSOTDataset',
        test_load_ann=True,
        ann_file=data_root + 'lasot_test/lasot_test.json',
        img_prefix=data_root + 'lasot_test/',
        pipeline=test_pipeline,
        ref_img_sampler=None,
        test_mode=False),
    val=dict(
        type='LaSOTDataset',
        test_load_ann=True,
        ann_file=data_root + 'lasot_test/lasot_test.json',
        img_prefix=data_root + 'lasot_test/',
        pipeline=test_pipeline,
        ref_img_sampler=None,
        test_mode=True),
    test=dict(
        type='LaSOTDataset',
        test_load_ann=True,
        ann_file=data_root + 'lasot_test/lasot_test.json',
        img_prefix=data_root + 'lasot_test/',
        pipeline=test_pipeline,
        ref_img_sampler=None,
        test_mode=True))
# optimizer
optimizer = dict(
    type='SGD',
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys=dict(backbone=dict(lr_mult=0.1, decay_mult=1.0))))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[3, 5])
# checkpoint saving
checkpoint_config = dict(interval=1)
evaluation = dict(interval=200)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 20
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/xxx'
load_from = None
resume_from = None
workflow = [('train', 1)]
