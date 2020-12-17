# Skip Review
# difference:
# use 1. two reg loss L1 or smooth L1
# use 2. roi align
# 3. img RGB or BGR
# use 4. bbox target_stds
# use 5. 3e2e1e or 25e35e05e
# use 6. train ref img sample -9--0 or -9--9
model = dict(
    type='FGFA',
    pretrains=dict(
        motion='data/imagenet_vid/pretrained_flownet/flownet_simple.pth'),
    detector=dict(
        type='FasterRCNN',
        pretrained='torchvision://resnet101',
        backbone=dict(
            type='ResNet',
            depth=101,
            num_stages=4,
            out_indices=(3, ),
            strides=(1, 2, 2, 1),
            dilations=(1, 1, 1, 2),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            style='pytorch'),
        neck=dict(
            type='ChannelMapper',
            in_channels=[2048],
            out_channels=512,
            kernel_size=3),
        rpn_head=dict(
            type='RPNHead',
            in_channels=512,
            feat_channels=512,
            anchor_generator=dict(
                type='AnchorGenerator',
                scales=[4, 8, 16, 32],
                ratios=[0.5, 1.0, 2.0],
                strides=[16]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[.0, .0, .0, .0],
                target_stds=[1.0, 1.0, 1.0, 1.0]),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            loss_bbox=dict(
                type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
        roi_head=dict(
            type='StandardRoIHead',
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign', output_size=7, sampling_ratio=2),
                out_channels=512,
                featmap_strides=[16]),
            bbox_head=dict(
                type='Shared2FCBBoxHead',
                in_channels=512,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=30,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.2, 0.2, 0.2, 0.2]),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0))),
        # detector training and testing settings
        train_cfg=dict(
            rpn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False),
                allowed_border=0,
                pos_weight=-1,
                debug=False),
            rpn_proposal=dict(
                nms_across_levels=False,
                nms_pre=6000,
                nms_post=600,
                max_num=600,
                nms_thr=0.7,
                min_bbox_size=0),
            rcnn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=256,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)),
        test_cfg=dict(
            rpn=dict(
                nms_across_levels=False,
                nms_pre=6000,
                nms_post=300,
                max_num=300,
                nms_thr=0.7,
                min_bbox_size=0),
            rcnn=dict(
                score_thr=0.0001,
                nms=dict(type='nms', iou_threshold=0.5),
                max_per_img=100))
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ),
    motion=dict(type='FlowNetSimple', img_scale_factor=0.5),
    aggregator=dict(
        type='EmbedAggregator', num_convs=1, channels=512, kernel_size=3),
)

# dataset settings
dataset_type = 'ImagenetVIDDataset'
data_root = 'data/imagenet_vid/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadMultiImagesFromFile'),
    dict(type='SeqLoadAnnotations', with_bbox=True, with_track=True),
    dict(type='SeqResize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
    dict(type='SeqNormalize', **img_norm_cfg),
    dict(type='SeqPad', size_divisor=16),
    dict(
        type='VideoCollect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_instance_ids']),
    dict(type='ConcatVideoReferences'),
    dict(type='SeqDefaultFormatBundle', ref_prefix='ref')
]
test_pipeline = [
    dict(type='LoadMultiImagesFromFile'),
    dict(type='SeqResize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.0),
    dict(type='SeqNormalize', **img_norm_cfg),
    dict(type='SeqPad', size_divisor=16),
    dict(
        type='VideoCollect',
        keys=['img'],
        meta_keys=('num_left_ref_imgs', 'frame_stride')),
    dict(type='ConcatVideoReferences'),
    dict(type='MultiImagesToTensor', ref_prefix='ref'),
    dict(type='ToList')
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=[
        dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/imagenet_vid_train.json',
            img_prefix=data_root + 'data/VID/',
            ref_img_sampler=dict(
                num_ref_imgs=2,
                frame_range=9,
                filter_key_img=True,
                method='bilateral_uniform'),
            pipeline=train_pipeline),
        dict(
            type=dataset_type,
            load_as_video=False,
            ann_file=data_root + 'annotations/imagenet_det_30plus1cls.json',
            img_prefix=data_root + 'data/DET/',
            ref_img_sampler=dict(
                num_ref_imgs=2,
                frame_range=0,
                filter_key_img=True,
                method='bilateral_uniform'),
            pipeline=train_pipeline)
    ],
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/imagenet_vid_val.json',
        img_prefix=data_root + 'data/VID/',
        ref_img_sampler=dict(
            num_ref_imgs=30,
            frame_range=[-15, 15],
            stride=1,
            method='test_with_fix_stride'),
        pipeline=test_pipeline,
        test_mode=True),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/imagenet_vid_val.json',
        img_prefix=data_root + 'data/VID/',
        ref_img_sampler=dict(
            num_ref_imgs=30,
            frame_range=[-15, 15],
            stride=1,
            method='test_with_fix_stride'),
        pipeline=test_pipeline,
        test_mode=True))
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[2, 5])
# checkpoint saving
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 7
dist_params = dict(backend='nccl', port='29500')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
evaluation = dict(metric=['bbox'], interval=7)
