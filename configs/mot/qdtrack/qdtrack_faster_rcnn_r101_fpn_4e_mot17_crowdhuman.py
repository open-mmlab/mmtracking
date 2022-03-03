_base_ = './qdtrack_faster_rcnn_r50_fpn.py'

model = dict(
    detector=dict(
        backbone=dict(norm_cfg=dict(requires_grad=False), style='caffe'),
        rpn_head=dict(bbox_coder=dict(clip_border=False)),
        roi_head=dict(
            bbox_head=dict(bbox_coder=dict(
                clip_border=False), num_classes=1))),
    track_head=dict(train_cfg=dict(assigner=dict(neg_iou_thr=0.5))))

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadMultiImagesFromFile', to_float32=True),
    dict(type='SeqLoadAnnotations', with_bbox=True, with_track=True),
    dict(
        type='SeqResize',
        img_scale=(1088, 1088),
        share_params=True,
        ratio_range=(0.8, 1.2),
        keep_ratio=True,
        bbox_clip_border=False),
    dict(type='SeqPhotoMetricDistortion', share_params=True),
    dict(
        type='SeqRandomCrop',
        share_params=False,
        crop_size=(1088, 1088),
        bbox_clip_border=False),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
    dict(type='SeqNormalize', **img_norm_cfg),
    dict(type='SeqPad', size_divisor=32),
    dict(type='MatchInstances', skip_nomatch=True),
    dict(
        type='VideoCollect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_match_indices'],
    ),
    dict(type='SeqDefaultFormatBundle', ref_prefix='ref')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1088, 1088),
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
mot_cfg = dict(
    type='MOTChallengeDataset',
    classes=('pedestrian', ),
    visibility_thr=-1,
    # track_visibility_thr=-1,
    ann_file='data/MOT17/annotations/half-train_cocoformat.json',
    img_prefix='data/MOT17/train',
    ref_img_sampler=dict(num_ref_imgs=1, frame_range=10, method='uniform'),
    pipeline=train_pipeline)
crowdhuman_cfg = dict(
    type='CocoVideoDataset',
    load_as_video=False,
    classes=('pedestrian', ),
    ann_file='data/crowdhuman/annotations/crowdhuman_train.json',
    img_prefix='data/crowdhuman/train',
    pipeline=train_pipeline)

dataset_type = 'MOTChallengeDataset'
data_root = 'data/MOT17/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='ConcatDataset',
        datasets=[mot_cfg, crowdhuman_cfg],
        saparate_eval=False),
    val=dict(
        type=dataset_type,
        ann_file='data/MOT17/annotations/half-val_cocoformat.json',
        classes=['pedestrian'],
        img_prefix='data/MOT17/train/',
        ref_img_sampler=None,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file='data/MOT17/annotations/half-val_cocoformat.json',
        img_prefix='data/MOT17/train/',
        classes=['pedestrian'],
        ref_img_sampler=None,
        pipeline=test_pipeline))

optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(policy='step', step=[3])
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50, hooks=[
        dict(type='TextLoggerHook'),
    ])
total_epochs = 4
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'ckpts/converted_faster_rcnn_r50_caffe_fpn_person_ap551.pth'
resume_from = None
workflow = [('train', 1)]
evaluation = dict(metric=['bbox', 'track'], interval=1)
work_dir = 'work_dirs/mix'
