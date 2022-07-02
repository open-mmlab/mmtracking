# dataset settings
train_pipeline = [
    dict(
        type='TransformBroadcaster',
        share_random_params=True,
        transforms=[
            dict(type='LoadImageFromFile'),
            dict(
                type='LoadTrackAnnotations',
                with_instance_id=True,
                with_mask=True,
                with_bbox=True),
            dict(type='mmdet.Resize', scale=(640, 360), keep_ratio=True),
            dict(type='mmdet.RandomFlip', prob=0.5),
        ]),
    dict(type='PackTrackInputs', ref_prefix='ref', num_key_frames=1)
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadTrackAnnotations',
        with_instance_id=True,
        with_mask=True,
        with_bbox=True),
    dict(type='mmdet.Resize', scale=(640, 360), keep_ratio=True),
    dict(type='PackTrackInputs', pack_single_img=True)
]

dataset_type = 'YouTubeVISDataset'
data_root = 'data/youtube_vis_2019/'
dataset_version = data_root[-5:-1]  # 2019 or 2021
# dataloader
train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='mmdet.AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        dataset_version=dataset_version,
        ann_file='annotations/youtube_vis_2019_train.json',
        data_prefix=dict(img_path='train/JPEGImages'),
        pipeline=train_pipeline,
        load_as_video=True,
        ref_img_sampler=dict(
            num_ref_imgs=1,
            frame_range=100,
            filter_key_img=True,
            method='uniform')))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='VideoSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        dataset_version=dataset_version,
        ann_file='annotations/youtube_vis_2019_valid.json',
        data_prefix=dict(img_path='valid/JPEGImages'),
        pipeline=test_pipeline,
        load_as_video=True,
        ref_img_sampler=None,
        test_mode=True))
test_dataloader = val_dataloader
