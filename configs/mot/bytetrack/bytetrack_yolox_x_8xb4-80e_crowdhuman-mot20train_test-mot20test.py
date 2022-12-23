_base_ = [
    './bytetrack_yolox_x_8xb4-80e_crowdhuman-mot17halftrain_'
    'test-mot17halfval.py'
]

dataset_type = 'MOTChallengeDataset'

img_scale = (896, 1600)

model = dict(
    data_preprocessor=dict(
        type='TrackDataPreprocessor',
        pad_size_divisor=32,
        batch_augments=[
            dict(
                type='mmdet.BatchSyncRandomResize',
                random_size_range=(640, 1152))
        ]),
    tracker=dict(
        weight_iou_with_det_scores=False,
        match_iou_thrs=dict(high=0.3),
    ))

train_pipeline = [
    dict(
        type='mmdet.Mosaic',
        img_scale=img_scale,
        pad_val=114.0,
        bbox_clip_border=True),
    dict(
        type='mmdet.RandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        bbox_clip_border=True),
    dict(
        type='mmdet.MixUp',
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0,
        bbox_clip_border=True),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.Resize',
        scale=img_scale,
        keep_ratio=True,
        clip_object_border=True),
    dict(
        type='mmdet.Pad',
        size_divisor=32,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(
        type='mmdet.FilterAnnotations',
        min_gt_bbox_wh=(1, 1),
        keep_empty=False),
    dict(type='PackTrackInputs', pack_single_img=True)
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='mmdet.Pad',
        size_divisor=32,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='PackTrackInputs', pack_single_img=True)
]

train_dataloader = dict(
    dataset=dict(
        type='mmdet.MultiImageMixDataset',
        dataset=dict(
            type='mmdet.ConcatDataset',
            datasets=[
                dict(
                    type='mmdet.CocoDataset',
                    data_root='data/MOT20',
                    ann_file='annotations/train_cocoformat.json',
                    # TODO: mmdet use img as key, but img_path is needed
                    data_prefix=dict(img='train'),
                    filter_cfg=dict(filter_empty_gt=True, min_size=32),
                    metainfo=dict(classes=('pedestrian')),
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadTrackAnnotations'),
                    ]),
                dict(
                    type='mmdet.CocoDataset',
                    data_root='data/crowdhuman',
                    ann_file='annotations/crowdhuman_train.json',
                    data_prefix=dict(img='train'),
                    filter_cfg=dict(filter_empty_gt=True, min_size=32),
                    metainfo=dict(classes=('pedestrian')),
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadTrackAnnotations'),
                    ]),
                dict(
                    type='mmdet.CocoDataset',
                    data_root='data/crowdhuman',
                    ann_file='annotations/crowdhuman_val.json',
                    data_prefix=dict(img='val'),
                    filter_cfg=dict(filter_empty_gt=True, min_size=32),
                    metainfo=dict(classes=('pedestrian')),
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadTrackAnnotations'),
                    ]),
            ]),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='VideoSampler'),
    dataset=dict(
        type=dataset_type,
        data_root='data/MOT17',
        ann_file='annotations/train_cocoformat.json',
        data_prefix=dict(img_path='train'),
        ref_img_sampler=None,
        load_as_video=True,
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='VideoSampler'),
    dataset=dict(
        type=dataset_type,
        data_root='data/MOT20',
        ann_file='annotations/test_cocoformat.json',
        data_prefix=dict(img_path='test'),
        ref_img_sampler=None,
        load_as_video=True,
        test_mode=True,
        pipeline=test_pipeline))

test_evaluator = dict(
    type='MOTChallengeMetrics',
    postprocess_tracklet_cfg=[
        dict(type='InterpolateTracklets', min_num_frames=5, max_num_frames=20)
    ],
    format_only=True,
    outfile_prefix='./mot_20_test_res')
