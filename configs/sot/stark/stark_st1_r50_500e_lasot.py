_base_ = ['./stark_st1_r50_500e_got10k.py']

data_root = 'data/'
train_pipeline = [
    dict(
        type='TridentSampling',
        num_search_frames=1,
        num_template_frames=2,
        max_frame_range=[200],
        cls_pos_prob=0.5,
        train_cls_head=False),
    dict(type='LoadMultiImagesFromFile', to_float32=True),
    dict(type='SeqLoadAnnotations', with_bbox=True, with_label=False),
    dict(type='SeqGrayAug', prob=0.05),
    dict(
        type='SeqRandomFlip',
        share_params=True,
        flip_ratio=0.5,
        direction='horizontal'),
    dict(
        type='SeqBboxJitter',
        center_jitter_factor=[0, 0, 4.5],
        scale_jitter_factor=[0, 0, 0.5],
        crop_size_factor=[2, 2, 5]),
    dict(
        type='SeqCropLikeStark',
        crop_size_factor=[2, 2, 5],
        output_size=[128, 128, 320]),
    dict(type='SeqBrightnessAug', jitter_range=0.2),
    dict(
        type='SeqRandomFlip',
        share_params=False,
        flip_ratio=0.5,
        direction='horizontal'),
    dict(
        type='SeqNormalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='CheckPadMaskValidity', stride=16),
    dict(
        type='VideoCollect',
        keys=['img', 'gt_bboxes', 'padding_mask'],
        meta_keys=('valid')),
    dict(type='ConcatSameTypeFrames', num_key_frames=2),
    dict(type='SeqDefaultFormatBundle', ref_prefix='search')
]

# dataset settings
data = dict(
    train=dict(
        type='RandomSampleConcatDataset',
        dataset_sampling_weights=[1, 1, 1, 1],
        dataset_cfgs=[
            dict(
                type='GOT10kDataset',
                img_prefix=data_root + 'got10k',
                pipeline=train_pipeline,
                split='train',
                test_mode=False),
            dict(
                type='LaSOTDataset',
                ann_file='tools/convert_datasets/lasot/testing_set.txt',
                img_prefix=data_root + 'lasot/LaSOTBenchmark',
                pipeline=train_pipeline,
                split='train',
                test_mode=False),
            dict(
                type='TrackingNetDataset',
                img_prefix=data_root + 'trackingnet',
                pipeline=train_pipeline,
                split='train',
                test_mode=False),
            dict(
                type='SOTCocoDataset',
                ann_file=data_root +
                'coco/annotations/instances_train2017.json',
                img_prefix=data_root + 'coco/train2017',
                pipeline=train_pipeline,
                split='train',
                test_mode=False)
        ]))
