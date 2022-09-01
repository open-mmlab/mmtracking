_base_ = ['../../_base_/default_runtime.py']

randomness = dict(seed=1, deterministic=True)
find_unused_parameters = True
crop_size = 511
exemplar_size = 127
search_size = 255

# model settings
model = dict(
    type='SiamRPN',
    data_preprocessor=dict(type='TrackDataPreprocessor'),
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
        type='mmdet.ChannelMapper',
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
            type='mmdet.DeltaXYWHBBoxCoder',
            target_means=[0., 0., 0., 0.],
            target_stds=[1., 1., 1., 1.]),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss', reduction='sum', loss_weight=1.0),
        loss_bbox=dict(type='mmdet.L1Loss', reduction='sum', loss_weight=1.2)),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='mmdet.MaxIoUAssigner',
                pos_iou_thr=0.6,
                neg_iou_thr=0.3,
                min_pos_iou=0.6,
                match_low_quality=False,
                iou_calculator=dict(type='mmdet.BboxOverlaps2D')),
            sampler=dict(
                type='mmdet.RandomSampler',
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

# data pipeline
data_root = 'data/'
train_pipeline = [
    dict(
        type='PairSampling',
        frame_range=100,
        pos_prob=0.8,
        filter_template_img=False),
    dict(
        type='TransformBroadcaster',
        share_random_params=False,
        transforms=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadTrackAnnotations', with_instance_id=False),
            dict(
                type='CropLikeSiamFC',
                context_amount=0.5,
                exemplar_size=exemplar_size,
                crop_size=crop_size)
        ]),
    dict(
        type='SeqShiftScaleAug',
        target_size=[exemplar_size, search_size],
        shift=[4, 64],
        scale=[0.05, 0.18]),
    dict(type='SeqColorAug', prob=[1.0, 1.0]),
    dict(type='SeqBlurAug', prob=[0.0, 0.2]),
    dict(type='PackTrackInputs', ref_prefix='search', num_template_frames=1)
]

# dataloader
train_dataloader = dict(
    batch_size=28,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='QuotaSampler', samples_per_epoch=600000),
    dataset=dict(
        type='RandomSampleConcatDataset',
        dataset_sampling_weights=[0.25, 0.2, 0.55],
        datasets=[
            dict(
                type='SOTImageNetVIDDataset',
                data_root=data_root,
                ann_file='ILSVRC/annotations/imagenet_vid_train.json',
                data_prefix=dict(img_path='ILSVRC/Data/VID'),
                pipeline=train_pipeline,
                test_mode=False),
            dict(
                type='SOTCocoDataset',
                data_root=data_root,
                ann_file='coco/annotations/instances_train2017.json',
                data_prefix=dict(img_path='coco/train2017'),
                pipeline=train_pipeline,
                test_mode=False),
            dict(
                type='SOTCocoDataset',
                data_root=data_root,
                ann_file='ILSVRC/annotations/imagenet_det_30plus1cls.json',
                data_prefix=dict(img_path='ILSVRC/Data/DET'),
                pipeline=train_pipeline,
                test_mode=False)
        ]))

# runner loop
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=20, val_begin=10, val_interval=1)

# learning policy
param_scheduler = [
    dict(
        type='SiamRPNExpLR',
        start_factor=0.2,
        end_factor=1.0,
        by_epoch=True,
        begin=0,
        end=5,
        endpoint=False),
    dict(
        type='SiamRPNExpLR',
        start_factor=1.0,
        end_factor=0.1,
        by_epoch=True,
        begin=5,
        end=20,
        endpoint=True)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=10.0, norm_type=2),
    paramwise_cfg=dict(
        custom_keys=dict(backbone=dict(lr_mult=0.1, decay_mult=1.0))))

custom_hooks = [
    dict(
        type='SiamRPNBackboneUnfreezeHook',
        backbone_start_train_epoch=10,
        backbone_train_layers=['layer2', 'layer3', 'layer4'])
]
