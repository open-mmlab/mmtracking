_base_ = ['./stark_r50_1x_got10k.py']
cudnn_benchmark = True
persistent_workers = True
crop_size = 511
exemplar_size = 127
search_size = 255

# model setting
model = dict(
    type='Stark',
    head=dict(
        type='StarkHead',
        run_bbox_head=False,
        run_cls_head=True,
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True)),
    frozen_modules=['backbone', 'neck'])

data_root = 'data/'
train_pipeline = [
    dict(type='LoadMultiImagesFromFile', to_float32=True),
    dict(type='SeqLoadAnnotations', with_bbox=True),
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
    dict(type='SeqBrightnessAug', brightness_jitter=0.2),
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
    dict(type='VideoCollect', keys=['img', 'gt_bboxes', 'att_mask']),
    dict(type='ConcatVideoTripleReferences'),
    dict(type='SeqDefaultFormatBundle', ref_prefix='search')
]
# dataset settings
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    persistent_workers=False,
    train=[
        dict(datasets_sampling_prob=[1], train_cls=True),
        dict(
            type='SOTQuotaTrainDataset',
            ann_file=data_root + 'got10k/annotations/got10k_train.json',
            img_prefix=data_root + 'got10k/train',
            pipeline=train_pipeline,
            max_gap=[200],
            num_search_frames=1,
            num_template_frames=2,
            cls_pos_prob=0.5,
            visible_keys=['absence', 'cover'],
            ref_img_sampler=None,
            test_mode=False),
        # dict(
        #     type='SOTQuotaTrainDataset',
        #     ann_file=data_root + 'lasot/annotations/lasot_train.json',
        #     img_prefix=data_root + 'lasot/LaSOTBenchmark',
        #     pipeline=train_pipeline,
        #     max_gap=[200],
        #     num_search_frames=1,
        #     num_template_frames=2,
        #     visible_keys=['full_occlusion', 'out_of_view'],
        #     ref_img_sampler=None,
        #     test_mode=False),
        # dict(
        #     type='SOTQuotaTrainDataset',
        #     ann_file=data_root +
        #     'trackingnet/annotations/trackingnet_train.json',
        #     img_prefix=data_root + 'trackingnet/train',
        #     pipeline=train_pipeline,
        #     max_gap=[200],
        #     num_search_frames=1,
        #     num_template_frames=2,
        #     visible_keys=None,
        #     ref_img_sampler=None,
        #     test_mode=False),
        # dict(
        #     type='SOTQuotaTrainDataset',
        #     ann_file=data_root + 'coco/annotations/instances_train2017.json',
        #     img_prefix=data_root + 'coco/train2017',
        #     pipeline=train_pipeline,
        #     max_gap=[200],
        #     num_search_frames=1,
        #     num_template_frames=2,
        #     visible_keys=None,
        #     ref_img_sampler=None,
        #     test_mode=False),
    ])

# learning policy
lr_config = dict(policy='step', step=[40])
# checkpoint saving
checkpoint_config = dict(interval=10)
evaluation = dict(
    metric=['track'],
    interval=10,
    start=50,
    rule='greater',
    save_best='success')
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 50
load_from = None
