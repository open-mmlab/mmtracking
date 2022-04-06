_base_ = ['./stark_st1_r50_500e_got10k.py']

# model setting
model = dict(
    type='Stark',
    head=dict(
        type='StarkHead',
        cls_head=dict(
            type='ScoreHead',
            input_dim=256,
            hidden_dim=256,
            output_dim=1,
            num_layers=3,
            use_bn=False),
        frozen_modules=['transformer', 'bbox_head', 'query_embedding'],
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True)),
    frozen_modules=['backbone', 'neck'])

data_root = 'data/'
train_pipeline = [
    dict(
        type='TridentSampling',
        num_search_frames=1,
        num_template_frames=2,
        max_frame_range=[200],
        cls_pos_prob=0.5,
        train_cls_head=True),
    dict(type='LoadMultiImagesFromFile', to_float32=True),
    dict(type='SeqLoadAnnotations', with_bbox=True, with_label=True),
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
        keys=['img', 'gt_bboxes', 'gt_labels', 'padding_mask'],
        meta_keys=('valid')),
    dict(type='ConcatSameTypeFrames', num_key_frames=2),
    dict(type='SeqDefaultFormatBundle', ref_prefix='search')
]

# dataset settings
data = dict(
    train=dict(dataset_cfgs=[
        dict(
            type='GOT10kDataset',
            ann_file=data_root + 'got10k/annotations/got10k_train_infos.txt',
            img_prefix=data_root + 'got10k',
            pipeline=train_pipeline,
            split='train',
            test_mode=False)
    ]))

# learning policy
lr_config = dict(policy='step', step=[40])
# checkpoint saving
checkpoint_config = dict(interval=10)
evaluation = dict(interval=100, start=51)
# yapf:enable
# runtime settings
total_epochs = 50
load_from = 'logs/stark_st1_got10k_online/epoch_500.pth'
