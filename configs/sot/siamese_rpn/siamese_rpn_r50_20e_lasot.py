cudnn_benchmark = False
deterministic = True
seed = 1
find_unused_parameters = True
crop_size = 511
exemplar_size = 127
search_size = 255

# model settings
model = dict(
    type='SiamRPN',
    backbone=dict(
        type='SOTResNet',
        depth=50,
        out_indices=(1, 2, 3),
        frozen_stages=4,
        strides=(1, 2, 1, 1),
        dilations=(1, 1, 2, 4),
        norm_eval=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://download.openmmlab.com/mmtracking/pretrained_weights/sot_resnet50.model'  # noqa: E501
        )),
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
        loss_bbox=dict(type='L1Loss', reduction='sum', loss_weight=1.2)),
    train_cfg=dict(
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
            search_size=search_size)),
    test_cfg=dict(
        exemplar_size=exemplar_size,
        search_size=search_size,
        context_amount=0.5,
        center_size=7,
        rpn=dict(penalty_k=0.05, window_influence=0.42, lr=0.38)))

data_root = 'data/'
train_pipeline = [
    dict(
        type='PairSampling',
        frame_range=100,
        pos_prob=0.8,
        filter_template_img=False),
    dict(type='LoadMultiImagesFromFile', to_float32=True),
    dict(type='SeqLoadAnnotations', with_bbox=True, with_label=False),
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
    dict(type='VideoCollect', keys=['img', 'gt_bboxes', 'is_positive_pairs']),
    dict(type='ConcatSameTypeFrames'),
    dict(type='SeqDefaultFormatBundle', ref_prefix='search')
]
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_label=False),
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
    workers_per_gpu=4,
    persistent_workers=True,
    samples_per_epoch=600000,
    train=dict(
        type='RandomSampleConcatDataset',
        dataset_sampling_weights=[0.25, 0.2, 0.55],
        dataset_cfgs=[
            dict(
                type='SOTImageNetVIDDataset',
                ann_file=data_root +
                'ILSVRC/annotations/imagenet_vid_train.json',
                img_prefix=data_root + 'ILSVRC/Data/VID',
                pipeline=train_pipeline,
                split='train',
                test_mode=False),
            dict(
                type='SOTCocoDataset',
                ann_file=data_root +
                'coco/annotations/instances_train2017.json',
                img_prefix=data_root + 'coco/train2017',
                pipeline=train_pipeline,
                split='train',
                test_mode=False),
            dict(
                type='SOTCocoDataset',
                ann_file=data_root +
                'ILSVRC/annotations/imagenet_det_30plus1cls.json',
                img_prefix=data_root + 'ILSVRC/Data/DET',
                pipeline=train_pipeline,
                split='train',
                test_mode=False)
        ]),
    val=dict(
        type='LaSOTDataset',
        ann_file=data_root + 'lasot/annotations/lasot_test_infos.txt',
        img_prefix=data_root + 'lasot/LaSOTBenchmark',
        pipeline=test_pipeline,
        split='test',
        test_mode=True,
        only_eval_visible=True),
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
evaluation = dict(
    metric=['track'],
    interval=1,
    start=10,
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
total_epochs = 20
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/xxx'
load_from = None
resume_from = None
workflow = [('train', 1)]
