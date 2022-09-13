_base_ = ['../../_base_/default_runtime.py']

randomness = dict(seed=1, deterministic=False)

# model setting
model = dict(
    type='PrDiMP',
    data_preprocessor=dict(
        type='TrackDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='mmcls.ResNet',
        depth=50,
        num_stages=3,
        strides=(1, 2, 2),
        dilations=[1, 1, 1],
        out_indices=[1, 2],  # 0, 1, 2
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    cls_head=dict(
        type='PrDiMPClsHead',
        in_dim=1024,
        out_dim=512,
        filter_initializer=dict(
            type='FilterInitializer',
            filter_size=4,
            feature_dim=512,
            feature_stride=16),
        filter_optimizer=dict(
            type='PrDiMPFilterOptimizer',
            num_iters=5,
            feat_stride=16,
            init_step_length=1.0,
            init_filter_regular=0.05,
            gauss_sigma=0.9,
            alpha_eps=0.05,
            min_filter_regular=0.05,
            label_thres=0),
        loss_cls=dict(type='KLGridLoss'),
        locate_cfg=dict(
            no_target_min_score=0.04,
            distractor_thres=0.8,
            hard_neg_thres=0.5,
            target_neighborhood_scale=2.2,
            dispalcement_scale=0.8,
            update_scale_when_uncertain=True),
        update_cfg=dict(
            sample_memory_size=50,
            normal_lr=0.01,
            hard_neg_lr=0.02,
            init_samples_min_weight=0.25,
            train_skipping=20),
        optimizer_cfg=dict(
            init_update_iters=10, update_iters=2, hard_neg_iters=1),
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
    bbox_head=dict(
        type='IouNetHead',
        in_dim=(4 * 128, 4 * 256),
        pred_in_dim=(256, 256),
        pred_inter_dim=(256, 256),
        loss_bbox=dict(type='KLMCLoss'),
        bbox_cfg=dict(
            num_init_random_boxes=9,
            box_jitter_pos=0.1,
            box_jitter_sz=0.5,
            iounet_topk=3,
            box_refine_step_length=2.5e-3,
            box_refine_iter=10,
            max_aspect_ratio=6,
            box_refine_step_decay=1),
        train_cfg=dict(
            proposals_sigma=[(0.05, 0.05), (0.5, 0.5)],
            gt_bboxes_sigma=(0.05, 0.05),
            num_samples=128,
            add_first_bbox=False,
            loss_weights=dict(bbox=0.0025))),
    test_cfg=dict(
        img_sample_size=22 * 16,
        feature_stride=16,
        search_scale_factor=6,
        patch_max_scale_change=1.5,
        border_mode='inside_major',
        bbox_inside_ratio=0.2,
        init_aug_cfg=dict(
            augmentation=dict(
                fliplr=True,
                rotate=[10, -10, 45, -45],
                blur=[(3, 1), (1, 3), (2, 2)],
                relativeshift=[(0.6, 0.6), (-0.6, 0.6), (0.6, -0.6),
                               (-0.6, -0.6)],
                dropout=[0.2, 0.2]),
            aug_expansion_factor=2,
            random_shift_factor=1 / 3)))

train_pipeline = [
    dict(
        type='DiMPSampling',
        num_search_frames=3,
        num_template_frames=3,
        max_frame_range=200),
    dict(
        type='TransformBroadcaster',
        share_random_params=True,
        transforms=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(type='LoadTrackAnnotations', with_instance_id=False),
            dict(type='GrayAug', prob=0.05)
        ]),
    dict(
        type='SeqBboxJitter',
        center_jitter_factor=[3, 3, 3, 4.5, 4.5, 4.5],
        scale_jitter_factor=[0.25, 0.25, 0.25, 0.5, 0.5, 0.5],
        crop_size_factor=[5, 5, 5, 5, 5, 5]),
    dict(
        type='TransformBroadcaster',
        share_random_params=False,
        transforms=[
            dict(type='CropLikeDiMP', crop_size_factor=5, output_size=288),
            dict(type='BrightnessAug', jitter_range=0.2)
        ]),
    dict(type='PackTrackInputs', ref_prefix='search', num_template_frames=3)
]

data_root = 'data/'
# dataset settings
train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='QuotaSampler', samples_per_epoch=60000),
    dataset=dict(
        type='RandomSampleConcatDataset',
        dataset_sampling_weights=[1, 1, 1, 1],
        datasets=[
            dict(
                type='GOT10kDataset',
                data_root=data_root,
                ann_file='GOT10k/annotations/got10k_train_vot_infos.txt',
                data_prefix=dict(img_path='GOT10k'),
                pipeline=train_pipeline,
                test_mode=False),
            dict(
                type='LaSOTDataset',
                data_root=data_root,
                ann_file='LaSOT_full/annotations/lasot_train_infos.txt',
                data_prefix=dict(img_path='LaSOT_full/LaSOTBenchmark'),
                pipeline=train_pipeline,
                test_mode=False),
            dict(
                type='TrackingNetDataset',
                chunks_list=[0, 1, 2, 3],
                data_root=data_root,
                ann_file='TrackingNet/annotations/trackingnet_train_infos.txt',
                data_prefix=dict(img_path='TrackingNet'),
                pipeline=train_pipeline,
                test_mode=False),
            dict(
                type='SOTCocoDataset',
                data_root=data_root,
                ann_file='coco/annotations/instances_train2017.json',
                data_prefix=dict(img_path='coco/train2017'),
                pipeline=train_pipeline,
                test_mode=False)
        ]))

# runner loop
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=50, val_begin=50, val_interval=1)

# learning policy
param_scheduler = dict(type='StepLR', step_size=15, gamma=0.2)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=2e-4),
    paramwise_cfg=dict(
        custom_keys=dict(
            backbone=dict(lr_multi=0.1),
            classifier=dict(lr_multi=5),
            bbox_regressor=dict(lr_multi=5))))

# checkpoint saving
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=10),
    logger=dict(type='LoggerHook', interval=50))
