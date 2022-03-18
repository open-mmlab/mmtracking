# cudnn_benchmark = False
# deterministic = True
# seed = 1

# model setting
model = dict(
    type='Prdimp',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=3,
        strides=(1, 2, 2),
        dilations=[1, 1, 1],
        out_indices=[1, 2],  # 0, 1, 2, 3
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='torchvision://resnet50'
        )),
    cls_head=dict(
        type='PrdimpClsHead',
        feat_dim=1024,
        out_dim=512,
        filter_initializer=dict(
            type='FilterClassifierInitializer',
            filter_size=4,
            feature_dim=512,
            feature_stride=16,
            pool_square=False,
            filter_norm=False,
            conv_ksz=3,
            init_weights_type='zero'),
        filter_optimizer=dict(
            type='PrDiMPSteepestDescentNewton',
            num_iter=5,
            feat_stride=16,
            init_step_length=1.0,
            init_filter_reg=0.05,
            gauss_sigma=0.9,
            detach_length=float('Inf'),
            alpha_eps=0.05,
            min_filter_reg=0.05,
            normalize_label=True,
            init_uni_weight=None,
            label_shrink=0,
            softmax_reg=None,
            label_threshold=0),
        loss_cls=dict(type='KLGridLoss'),
        locate_cfg=dict(
            score_preprocess='softmax',
            target_not_found_threshold=0.04,
            distractor_threshold=0.8,
            hard_negative_threshold=0.5,
            target_neighborhood_scale=2.2,
            dispalcement_scale=0.8,
            hard_negative_learning_rate=0.02,
            update_scale_when_uncertain=True),
        update_cfg=dict(
            sample_memory_size=50,
            learning_rate=0.01,
            init_samples_minimum_weight=0.25,
            train_skipping=20),
        optimizer_cfg=dict(
            net_opt_iter=10, net_opt_update_iter=2, net_opt_hn_iter=1),
        train_cfg=dict(
            feat_size=(18, 18),
            img_size=(288, 288),
            sigma_factor=0.05,
            end_pad_if_even=True,
            gauss_label_bias=0.,
            use_gauss_density=True,
            label_density_norm=True,
            label_density_threshold=0.,
            label_density_shrink=0,
            loss_weights=dict(cls_init=0.25, cls_iter=1., cls_final=0.25))),
    reg_head=dict(
        type='IouNetHead',
        input_dim=(4 * 128, 4 * 256),
        pred_input_dim=(256, 256),
        pred_inter_dim=(256, 256),
        loss_bbox=dict(type='KLMCLoss'),
        bbox_cfg=dict(
            num_init_random_boxes=9,
            box_jitter_pos=0.1,
            box_jitter_sz=0.5,
            iounet_k=3,
            box_refinement_step_length=2.5e-3,
            box_refinement_iter=10,
            maximal_aspect_ratio=6,
            box_refinement_step_decay=1),
        train_cfg=dict(
            proposals_sigma=[(0.05, 0.05), (0.5, 0.5)],
            gt_bboxes_sigma=(0.05, 0.05),
            num_samples=128,
            add_first_bbox=False,
            loss_weights=dict(bbox=0.0025))),
    test_cfg=dict(
        image_sample_size=22 * 16,
        feature_stride=16,
        search_area_scale=6,
        patch_max_scale_change=1.5,
        border_mode='inside_major',
        scale_factor=[1],
        init_aug_cfg=dict(
            augmentation=dict(
                fliplr=True,
                rotate=[10, -10, 45, -45],
                blur=[(3, 1), (1, 3), (2, 2)],
                relativeshift=[(0.6, 0.6), (-0.6, 0.6), (0.6, -0.6),
                               (-0.6, -0.6)],
                dropout=(2, 0.2)),
            augmentation_expansion_factor=2,
            random_shift_factor=1 / 3)))

file_client_args = dict(
    backend='petrel',
    path_mapping=dict({
        'data/got10k':
        'openmmlab:s3://openmmlab/datasets/tracking/GOT10k',
        'data/trackingnet':
        'openmmlab:s3://openmmlab/datasets/tracking/TrackingNet',
        'data/lasot':
        'openmmlab:s3://openmmlab/datasets/tracking/LaSOT_full',
        'data/coco':
        'openmmlab:s3://openmmlab/datasets/detection/coco'
    }))

train_pipeline = [
    dict(
        type='DimpSampling',
        num_search_frames=3,
        num_template_frames=3,
        max_frame_range=200),
    dict(
        type='LoadMultiImagesFromFile',
        to_float32=True,
        file_client_args=file_client_args),
    dict(
        type='SeqLoadAnnotations',
        with_bbox=True,
        with_label=False,
        file_client_args=file_client_args),
    dict(type='SeqGrayAug', prob=0.05),
    dict(
        type='SeqBboxJitter',
        center_jitter_factor=[3, 3, 3, 4.5, 4.5, 4.5],
        scale_jitter_factor=[0.25, 0.25, 0.25, 0.5, 0.5, 0.5],
        crop_size_factor=[5, 5, 5, 5, 5, 5]),
    dict(
        type='SeqCropLikeDimp',
        crop_size_factor=[5, 5, 5, 5, 5, 5],
        output_size=[288, 288, 288, 288, 288, 288]),
    dict(type='SeqBrightnessAug', jitter_range=0.2, share_params=False),
    dict(
        type='SeqNormalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='VideoCollect', keys=['img', 'gt_bboxes']),
    dict(type='ConcatSameTypeFrames', num_key_frames=3),
    dict(type='SeqDefaultFormatBundle', ref_prefix='search')
]

img_norm_cfg = dict(mean=[0, 0, 0], std=[1, 1, 1], to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_label=False),
    dict(
        type='MultiScaleFlipAug',
        scale_factor=1,
        flip=False,
        transforms=[
            dict(type='Normalize', **img_norm_cfg),
            dict(type='VideoCollect', keys=['img', 'gt_bboxes']),
            dict(type='ImageToTensor', keys=['img'])
        ])
]

data_root = 'data/'

# dataset settings
data = dict(
    samples_per_gpu=10,
    workers_per_gpu=8,
    persistent_workers=True,
    samples_per_epoch=26000,
    train=dict(
        type='RandomSampleConcatDataset',
        dataset_sampling_weights=[0.25,1,1],
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
                'coco/annotations/instances_train2014.json',
                img_prefix=data_root + 'coco/train2014',
                pipeline=train_pipeline,
                split='train',
                test_mode=False)
        ]),
    val=dict(
        type='GOT10kDataset',
        img_prefix=data_root + 'got10k',
        pipeline=test_pipeline,
        split='test',
        test_mode=True),
    test=dict(
        type='GOT10kDataset',
        img_prefix=data_root + 'got10k',
        pipeline=test_pipeline,
        split='test',
        test_mode=True))

# optimizer
optimizer = dict(
    type='Adam',
    lr=2e-4,
    weight_decay=0,
    paramwise_cfg=dict(
        custom_keys=dict(
            backbone=dict(lr_multi=0.1),
            cls_head=dict(lr_multi=5),
            bbox_head=dict(lr_multi=5))))
optimizer_config = dict(type='OptimizerHook')
# learning policy
lr_config = dict(policy='step', step=15, gamma=0.2)
# checkpoint saving
checkpoint_config = dict(
    interval=10,
    out_dir='sh1984:s3://zhangjingwei/mmtracking_others/mmtracking_0/prdimp')
evaluation = dict(
    metric=['track'],
    interval=10,
    start=51,
    rule='greater',
    save_best='success',
    out_dir='sh1984:s3://zhangjingwei/mmtracking_others/mmtracking_0/prdimp')
# yapf:disable
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook', out_dir='sh1984:s3://zhangjingwei/mmtracking_others/mmtracking_0/prdimp')  # noqa: E501
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 50
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/xxx'
load_from = None
resume_from = None
workflow = [('train', 1)]
