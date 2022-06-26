_base_ = [
    './qdtrack_faster-rcnn_r50_fpn_4e_base.py',
    '../../_base_/datasets/mot_challenge.py',
]
dataset_type = 'MOTChallengeDataset'
data_root = 'data/MOT17/'

# data pipeline
train_pipeline = [
    dict(
        type='TransformBroadcaster',
        share_random_params=True,
        transforms=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadTrackAnnotations', with_instance_id=True),
            dict(
                type='mmdet.Resize',
                scale=(1088, 1088),
                scale_factor=(0.8, 1.2),
                keep_ratio=True,
                clip_object_border=False),
            dict(type='mmdet.PhotoMetricDistortion')
        ]),
    dict(
        type='TransformBroadcaster',
        share_random_params=False,
        transforms=[
            dict(
                type='mmdet.RandomCrop',
                crop_size=(1088, 1088),
                bbox_clip_border=False)
        ]),
    dict(
        type='TransformBroadcaster',
        share_random_params=True,
        transforms=[
            dict(type='mmdet.RandomFlip', prob=0.5),
        ]),
    dict(type='PackTrackInputs', ref_prefix='ref')
]
# dataloader
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='mmdet.AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        visibility_thr=-1,
        ann_file='annotations/half-train_cocoformat.json',
        data_prefix=dict(img_path='train'),
        ref_img_sampler=dict(
            num_ref_imgs=1,
            frame_range=10,
            filter_key_img=True,
            method='uniform'),
        pipeline=train_pipeline))

# evaluator
val_evaluator = [
    dict(type='MOTChallengeMetrics', metric=['HOTA', 'CLEAR', 'Identity']),
    dict(type='CocoVideoMetric', metric=['bbox'])
]
test_evaluator = val_evaluator
