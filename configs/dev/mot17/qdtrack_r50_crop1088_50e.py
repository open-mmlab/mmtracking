_base_ = [
    '../../_base_/models/qdtrack_faster-rcnn_r50_fpn.py',
    '../../_base_/default_runtime.py'
]
# save_variables = ['det_bboxes', 'det_labels', 'embeds']
search_metrics = ['MOTA', 'IDF1', 'FN', 'FP', 'IDs']
model = dict(
    pretrains=dict(detector='ckpts/mmdet/faster_rcnn_r50_fpn_2x_coco_bbox' +
                   '_mAP-0.384_20200504_210434-a5d8aa15.pth'),
    frozen_modules=None,
    detector=dict(
        roi_head=dict(bbox_head=dict(num_classes=1)),
        test_cfg=dict(
            rcnn=dict(
                score_thr=0.05,
                nms=dict(type='nms', iou_threshold=0.5),
                max_per_img=100))),
    tracker=dict(
        type='MOT17Tracker',
        init_score_thr=0.8,
        obj_score_thr=0.4,
        match_score_thr=0.5,
        memo_tracklet_frames=30,
        memo_backdrop_frames=1,
        memo_momentum=0.8,
        nms_conf_thr=0.5,
        nms_backdrop_iou_thr=0.3,
        nms_class_iou_thr=0.7,
        with_cats=True,
        match_metric='bisoftmax',
        cosine_factor=0.0))
dataset_type = 'MOT17Dataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadMultiImagesFromFile', to_float32=True),
    dict(type='SeqLoadAnnotations', with_bbox=True, with_track=True),
    dict(
        type='SeqResize',
        img_scale=(1088, 1088),
        share_params=True,
        ratio_range=(0.8, 1.2),
        keep_ratio=True),
    dict(type='SeqPhotoMetricDistortion', share_params=True),
    dict(type='SeqRandomCrop', share_params=False, crop_size=(1088, 1088)),
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
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        visibility_thr=0.1,
        ann_file='data/mot17/annotations/mot17_train_cocoformat.json',
        img_prefix='data/mot17/images/train/',
        ref_img_sampler=dict(
            num_ref_imgs=1,
            frame_range=3,
            filter_key_img=True,
            method='uniform'),
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file='data/mot17/annotations/mot17_train_cocoformat.json',
        img_prefix='data/mot17/images/train/',
        ref_img_sampler=None,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file='data/mot17det/annotations/mot17_test_cocoformat.json',
        img_prefix='data/mot17det/test/',
        ref_img_sampler=None,
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[10, 12])
total_epochs = 14
evaluation = dict(
    metric=['bbox', 'track'],
    interval=1,
    resfile_path='/mnt/lustre/pangjiangmiao/results/mot/')
