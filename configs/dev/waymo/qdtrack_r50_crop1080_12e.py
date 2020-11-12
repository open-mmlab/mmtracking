_base_ = [
    '../../_base_/models/qdtrack_faster-rcnn_r50_fpn.py',
    '../../_base_/default_runtime.py'
]
model = dict(
    pretrains=dict(
        detector='ckpts/mmdet/faster_rcnn_r50_fpn_2x_coco_bbox_mAP' +
        '-0.384_20200504_210434-a5d8aa15.pth'),
    frozen_modules=None,
    detector=dict(roi_head=dict(bbox_head=dict(num_classes=3))))
dataset_type = 'CocoVideoDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadMultiImagesFromFile'),
    dict(type='SeqLoadAnnotations', with_bbox=True, with_track=True),
    dict(
        type='SeqResize',
        img_scale=(1080, 1080),
        share_params=True,
        ratio_range=(0.8, 1.2),
        keep_ratio=True),
    dict(type='SeqRandomCrop', share_params=False, crop_size=(1080, 1080)),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
    dict(type='SeqNormalize', **img_norm_cfg),
    dict(type='SeqPad', size_divisor=32),
    dict(type='MatchInstances', skip_nomatch=True),
    dict(
        type='VideoCollect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_match_indices']),
    dict(type='SeqDefaultFormatBundle', ref_prefix='ref')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1080, 1080),
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
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        classes=('vehicle', 'pedestrian', 'cyclist'),
        ann_file='data/waymo/annotations/waymo12_all_train_3cls.json',
        img_prefix='data/waymo/images/',
        ref_img_sampler=dict(
            num_ref_imgs=1,
            frame_range=3,
            filter_key_img=True,
            method='uniform'),
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=('vehicle', 'pedestrian', 'cyclist'),
        ann_file='data/waymo/annotations/waymo12_all_val_3cls.json',
        img_prefix='data/waymo/images/',
        ref_img_sampler=None,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=('vehicle', 'pedestrian', 'cyclist'),
        ann_file='data/waymo/annotations/waymo12_all_val_3cls.json',
        img_prefix='data/waymo/images/',
        ref_img_sampler=None,
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=5000,
    warmup_ratio=0.001,
    step=[8, 11])
total_epochs = 12
evaluation = dict(metric=['bbox', 'track'], interval=12)
