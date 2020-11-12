_base_ = [
    '../../_base_/models/qdtrack_faster-rcnn_r50_fpn.py',
    '../../_base_/default_runtime.py'
]
search_metrics = ['MOTA', 'IDF1', 'FN', 'FP', 'IDs']
# save_variables = ['det_bboxes', 'det_labels', 'embeds']
model = dict(
    pretrains=dict(
        detector='ckpts/mmdet/faster_rcnn_r50_caffe_fpn_person_ap551.pth'),
    frozen_modules=None,
    detector=dict(
        pretrained=None,
        backbone=dict(norm_cfg=dict(requires_grad=False), style='caffe'),
        rpn_head=dict(bbox_coder=dict(clip_border=False)),
        roi_head=dict(
            bbox_head=dict(bbox_coder=dict(clip_border=False), num_classes=1)),
        test_cfg=dict(rcnn=dict(nms=dict(type='nms', iou_threshold=0.5)))),
    track_head=dict(
        roi_assigner=dict(neg_iou_thr=0.5),
        embed_head=dict(loss_track=dict(loss_weight=0.25))),
    tracker=dict(
        type='MOT17Tracker',
        init_score_thr=0.9,
        obj_score_thr=0.5,
        match_score_thr=0.5,
        memo_tracklet_frames=30,
        memo_backdrop_frames=1,
        memo_momentum=0.8,
        nms_conf_thr=0.5,
        nms_backdrop_iou_thr=0.3,
        nms_class_iou_thr=0.7,
        with_cats=True,
        match_metric='bisoftmax'))
dataset_type = 'MOT17Dataset'
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
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        visibility_thr=-1,
        track_visibility_thr=0.1,
        ann_file='data/mot17det/annotations/mot17_half-train_cocoformat.json',
        img_prefix='data/mot17det/train/',
        ref_img_sampler=dict(
            num_ref_imgs=1,
            frame_range=10,
            filter_key_img=True,
            method='uniform'),
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file='data/mot17det/annotations/mot17_half-val_cocoformat.json',
        img_prefix='data/mot17det/train/',
        ref_img_sampler=None,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file='data/mot17det/annotations/mot17_half-val_cocoformat.json',
        img_prefix='data/mot17det/train/',
        ref_img_sampler=None,
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[6])
total_epochs = 9
evaluation = dict(metric=['bbox', 'track'], interval=1)
checkpoint_config = dict(interval=1)
dist_params = dict(port='12349')
# log_config = dict(interval=1)
