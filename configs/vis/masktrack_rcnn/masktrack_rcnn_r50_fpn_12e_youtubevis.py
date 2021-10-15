_base_ = [
    '../../_base_/models/mask_rcnn_r50_fpn.py',
    '../../_base_/default_runtime.py'
]
model = dict(
    type='MaskTrackRCNN',
    detector=dict(
        roi_head=dict(
            bbox_head=dict(num_classes=40), mask_head=dict(num_classes=40)),
        train_cfg=dict(
            rpn=dict(sampler=dict(num=64)),
            rpn_proposal=dict(nms_pre=200, max_per_img=200),
            rcnn=dict(sampler=dict(num=128))),
        test_cfg=dict(
            rpn=dict(nms_pre=200, max_per_img=200), rcnn=dict(score_thr=0.01)),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'  # noqa: E501
        )),
    track_head=dict(
        type='RoITrackHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        embed_head=dict(
            type='RoIEmbedHead',
            num_fcs=2,
            roi_feat_size=7,
            in_channels=256,
            fc_out_channels=1024),
        train_cfg=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=128,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    # TODO: Support tracker
    # tracker=dict(
    #     type='MaskTrackRCNNTracker',
    #     score_coefficient=1.0,
    #     iou_coefficient=2.0,
    #     label_coefficient=10.0))
    tracker=None)

# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadMultiImagesFromFile', to_float32=True),
    dict(
        type='SeqLoadAnnotations',
        with_bbox=True,
        with_mask=True,
        with_track=True),
    dict(
        type='SeqResize',
        share_params=True,
        img_scale=(640, 360),
        keep_ratio=True),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
    dict(type='SeqNormalize', **img_norm_cfg),
    dict(type='SeqPad', size_divisor=32),
    dict(
        type='VideoCollect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_instance_ids']),
    dict(type='SeqDefaultFormatBundle', ref_prefix='ref')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 360),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='VideoCollect', keys=['img'])
        ])
]
dataset_type = 'YouTubeVISDataset'
data_root = 'data/youtube_vis_2019/'
dataset_version = data_root[-5:-1]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        dataset_version=dataset_version,
        ann_file=data_root + 'annotations/youtube_vis_2019_train.json',
        img_prefix=data_root + 'train/JPEGImages',
        ref_img_sampler=dict(
            num_ref_imgs=1,
            frame_range=100,
            filter_key_img=True,
            method='uniform'),
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        dataset_version=dataset_version,
        ann_file=data_root + 'annotations/youtube_vis_2019_valid.json',
        img_prefix=data_root + 'valid/JPEGImages',
        ref_img_sampler=None,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        dataset_version=dataset_version,
        ann_file=data_root + 'annotations/youtube_vis_2019_valid.json',
        img_prefix=data_root + 'valid/JPEGImages',
        ref_img_sampler=None,
        pipeline=test_pipeline))

# optimizer
optimizer = dict(type='SGD', lr=0.00125, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
# runtime settings
total_epochs = 12
evaluation = dict(metric=['track_segm'], interval=13)
