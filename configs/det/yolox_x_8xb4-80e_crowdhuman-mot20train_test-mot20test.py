_base_ = [
    './yolox_x_8xb4-80e_crowdhuman-mot17halftrain'
    '_test-mot17halfval.py'
]

img_scale = (896, 1600)

model = dict(
    data_preprocessor=dict(batch_augments=[
        dict(type='BatchSyncRandomResize', random_size_range=(640, 1152))
    ]))

train_pipeline = [
    dict(
        type='Mosaic',
        img_scale=img_scale,
        pad_val=114.0,
        bbox_clip_border=True),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        bbox_clip_border=True),
    dict(
        type='MixUp',
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0,
        bbox_clip_border=True),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='Resize',
        scale=img_scale,
        keep_ratio=True,
        clip_object_border=True),
    dict(type='Pad', size_divisor=32, pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='Pad', size_divisor=32, pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    dataset=dict(
        type='MultiImageMixDataset',
        _scope_='mmdet',
        dataset=dict(
            type='ConcatDataset',
            _scope_='mmdet',
            datasets=[
                dict(
                    type='CocoDataset',
                    data_root='data/MOT20',
                    ann_file='annotations/train_cocoformat.json',
                    data_prefix=dict(img='train'),
                    filter_cfg=dict(filter_empty_gt=True, min_size=32),
                    metainfo=dict(classes=('pedestrian', )),
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations'),
                    ]),
                dict(
                    type='CocoDataset',
                    data_root='data/crowdhuman',
                    ann_file='annotations/crowdhuman_train.json',
                    data_prefix=dict(img='train'),
                    filter_cfg=dict(filter_empty_gt=True, min_size=32),
                    metainfo=dict(classes=('pedestrian', )),
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations'),
                    ]),
                dict(
                    type='CocoDataset',
                    data_root='data/crowdhuman',
                    ann_file='annotations/crowdhuman_val.json',
                    data_prefix=dict(img='val'),
                    filter_cfg=dict(filter_empty_gt=True, min_size=32),
                    metainfo=dict(classes=('pedestrian', )),
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations'),
                    ]),
            ]),
        pipeline=train_pipeline))
val_dataloader = dict(
    dataset=dict(
        type='CocoDataset',
        data_root='data/MOT17',
        _scope_='mmdet',
        ann_file='annotations/train_cocoformat.json',
        data_prefix=dict(img='train'),
        metainfo=dict(classes=('pedestrian', )),
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = dict(
    dataset=dict(
        type='CocoDataset',
        data_root='data/MOT20',
        _scope_='mmdet',
        ann_file='annotations/test_cocoformat.json',
        data_prefix=dict(img='test'),
        metainfo=dict(classes=('pedestrian', )),
        test_mode=True,
        pipeline=test_pipeline))

val_evaluator = dict(
    type='mmdet.CocoMetric',
    ann_file='data/MOT17/annotations/train_cocoformat.json',
    metric='bbox',
    format_only=False)
test_evaluator = dict(
    type='mmdet.CocoMetric',
    ann_file='data/MOT20/annotations/test_cocoformat.json',
    metric='bbox',
    format_only=True)
