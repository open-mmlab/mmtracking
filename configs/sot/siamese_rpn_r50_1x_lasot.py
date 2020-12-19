# SKIP REVIEW
cudnn_benchmark = True

crop_size = 511
exemplar_size = 127
search_size = 255

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
        weighted_sum=True,
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0., 0., 0., 0.],
            target_stds=[1., 1., 1., 1.]),
        loss_cls=dict(
            type='CrossEntropyLoss', reduction='sum', loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', reduction='sum', loss_weight=1.2)))
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.6,
            neg_iou_thr=0.3,
            min_pos_iou=0.6,
            match_low_quality=False),
        sampler=dict(
            type='RandomSampler',
            num=64,
            pos_fraction=0.25,
            add_gt_as_proposals=False),
        num_neg=16,
        exemplar_size=exemplar_size,
        search_size=search_size))
test_cfg = dict(
    exemplar_size=exemplar_size,
    search_size=search_size,
    context_amount=0.5,
    center_size=7,
    rpn=dict(penalty_k=0.05, window_influence=0.42, lr=0.38))

data_root = 'data/sot/'
train_pipeline = [
    dict(type='LoadMultiImagesFromFile', to_float32=True),
    dict(type='SeqLoadAnnotations', with_bbox=True),
    dict(
        type='SeqCropLikeSiamFC',
        context_amount=0.5,
        exemplar_size=exemplar_size,
        crop_size=crop_size),
    dict(
        type='SeqShiftScaleAug',
        target_size=[exemplar_size, search_size],
        shift=[4, 64],
        scale=[0.05, 0.18]),
    dict(type='SeqColorAug', prob=[1.0, 1.0]),
    dict(type='SeqBlurAug', prob=[0.0, 0.2]),
    dict(type='VideoCollect', keys=['img', 'gt_bboxes', 'is_positive_pair']),
    dict(type='ConcatVideoReferences'),
    dict(type='SeqDefaultFormatBundle', ref_prefix='search')
]
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
# dataset settings
data = dict(
    samples_per_gpu=28,
    workers_per_gpu=2,
    train=[
        dict(
            type='RepeatDataset',
            times=39,
            dataset=dict(
                type='SOTTrainDataset',
                ann_file=data_root +
                'train/annotations/imagenet_vid/imagenet_vid_train.json',
                img_prefix=data_root + 'train/data/imagenet_vid',
                pipeline=train_pipeline,
                ref_img_sampler=dict(
                    frame_range=100,
                    pos_prob=0.8,
                    filter_key_img=False,
                    return_key_img=True),
            )),
        dict(
            type='SOTTrainDataset',
            ann_file=data_root +
            'train/annotations/coco2017_train/instances_train2017.json',
            img_prefix=data_root + 'train/data/coco2017_train',
            pipeline=train_pipeline,
            ref_img_sampler=dict(
                frame_range=0,
                pos_prob=0.8,
                filter_key_img=False,
                return_key_img=True),
        ),
        dict(
            type='SOTTrainDataset',
            ann_file=data_root +
            'train/annotations/imagenet_det/imagenet_det_30plus1cls.json',
            img_prefix=data_root + 'train/data/imagenet_det',
            pipeline=train_pipeline,
            ref_img_sampler=dict(
                frame_range=0,
                pos_prob=0.8,
                filter_key_img=False,
                return_key_img=True),
        ),
    ],
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
optimizer_config = dict(
    type='SiameseRPNOptimizerHook',
    backbone_start_train_epoch=10,
    backbone_train_layers=['layer2', 'layer3', 'layer4'],
    grad_clip=dict(max_norm=10.0, norm_type=2))
# learning policy
lr_config = dict(
    policy='SiameseRPN',
    lr_configs=[
        dict(type='step', start_lr_factor=0.2, end_lr_factor=1.0, end_epoch=5),
        dict(type='log', start_lr_factor=1.0, end_lr_factor=0.1, end_epoch=20),
    ])
# checkpoint saving
checkpoint_config = dict(interval=1)
evaluation = dict(interval=20)
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
