# dataset settings
dataset_type = 'TaoDataset'
data_root = 'data/tao/'

# data pipeline
train_pipeline = [
    dict(
        type='TransformBroadcaster',
        share_random_params=True,
        transforms=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadTrackAnnotations', with_instance_id=True),
            dict(
                type='RandomChoiceResize',
                scales=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                        (1333, 768), (1333, 800)],
                resize_type='mmdet.Resize',
                keep_ratio=True),
            dict(type='mmdet.RandomFlip', prob=0.5),
        ]),
    dict(type='PackTrackInputs', ref_prefix='ref')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadTrackAnnotations', with_instance_id=True),
    dict(type='mmdet.Resize', scale=(1333, 800), keep_ratio=True),
    dict(
        type='PackTrackInputs',
        pack_single_img=True,
        meta_keys=('frame_index', 'neg_category_ids',
                   'not_exhaustive_category_ids'))
]

# dataloader
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    batch_sampler=dict(type='mmdet.AspectRatioBatchSampler'),
    # TODO: wait mmdet.ClassBalancedDataset
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        load_as_video=False,
        metainfo=dict(CLASSES=data_root + 'annotations/tao_classes.txt'),
        ann_file='annotations/lvisv0.5+coco_train.json',
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='VideoSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dict(CLASSES=data_root + 'annotations/tao_classes.txt'),
        ann_file='annotations/validation_482_classes.json',
        data_prefix=dict(img_path='val'),
        ref_img_sampler=None,
        load_as_video=True,
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader
