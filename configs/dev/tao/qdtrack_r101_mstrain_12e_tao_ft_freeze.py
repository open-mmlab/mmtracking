_base_ = [
    '../../_base_/models/qdtrack_faster-rcnn_r50_fpn.py',
    '../../_base_/default_runtime.py'
]
save_variables = ['det_bboxes', 'det_labels', 'embeds']
model = dict(
    pretrains=None,
    frozen_modules='detector',
    detector=dict(
        pretrained='torchvision://resnet101',
        backbone=dict(depth=101),
        roi_head=dict(bbox_head=dict(num_classes=482)),
        test_cfg=dict(rcnn=dict(score_thr=0.0001, max_per_img=300))),
    tracker=dict(
        type='TaoTracker',
        init_score_thr=0.0001,
        obj_score_thr=0.0001,
        match_score_thr=0.5,
        memo_tracklet_frames=10,
        memo_backdrop_frames=0,
        memo_momentum=0.8,
        nms_conf_thr=1.0,
        nms_backdrop_iou_thr=1.0,
        nms_class_iou_thr=1.0,
        with_cats=True,
        match_metric='cosine'))
dataset_type = 'TaoDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadMultiImagesFromFile'),
    dict(type='SeqLoadAnnotations', with_bbox=True, with_track=True),
    dict(
        type='SeqResize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                   (1333, 768), (1333, 800)],
        share_params=True,
        multiscale_mode='value',
        keep_ratio=True),
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
        img_scale=(1333, 800),
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
    workers_per_gpu=2,
    train=dict(
        _delete_=True,
        type='ClassBalancedDataset',
        oversample_thr=1e-3,
        dataset=dict(
            type=dataset_type,
            classes='data/tao/annotations/tao_classes.txt',
            ann_file='data/tao/annotations/train_ours.json',
            img_prefix='data/tao/frames/',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        classes='data/tao/annotations/tao_classes.txt',
        ann_file='data/tao/annotations/validation_ours.json',
        img_prefix='data/tao/frames/',
        ref_img_sampler=None,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes='data/tao/annotations/tao_classes.txt',
        ann_file='data/tao/annotations/tao_mini_val.json',
        img_prefix='data/tao/frames/',
        ref_img_sampler=None,
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
total_epochs = 12
evaluation = dict(metric=['bbox', 'track'], interval=2)
load_from = 'work_dirs/dev/tao/qdtrack_r101_mstrain_2x_lvis/epoch_24.pth'
