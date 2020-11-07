_base_ = [
    '../../_base_/models/qdtrack_faster-rcnn_r50_fpn.py',
    '../../_base_/default_runtime.py'
]
model = dict(
    pretrains=dict(
        detector='ckpts/mmdet/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_' +
        'bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'),
    frozen_modules=None,
    detector=dict(
        pretrained=None,
        backbone=dict(norm_cfg=dict(requires_grad=False), style='caffe'),
        roi_head=dict(bbox_head=dict(num_classes=1))),
    track_head=dict(embed_head=dict(loss_track=dict(loss_weight=0.25))))
dataset_type = 'CocoVideoDataset'
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
        keep_ratio=True),
    # dict(type='SeqPhotoMetricDistortion', share_params=True),
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
data_root = 'data/coco/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        load_as_video=False,
        classes=('person', ),
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        ref_img_sampler=dict(frame_range=0),
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        load_as_video=False,
        classes=('person', ),
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        ref_img_sampler=None,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        load_as_video=False,
        classes=('person', ),
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        ref_img_sampler=None,
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.04, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2000,
    warmup_ratio=0.001,
    step=[8, 11])
total_epochs = 12
evaluation = dict(metric='bbox', interval=1)
checkpoint_config = dict(interval=1)
