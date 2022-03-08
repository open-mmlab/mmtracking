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
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
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
            init_weights='default'),
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
            net_opt_iter=10, net_opt_update_iter=2, net_opt_hn_iter=1)),
    reg_head=dict(
        type='IouNetHead',
        input_dim=(4 * 128, 4 * 256),
        pred_input_dim=(256, 256),
        pred_inter_dim=(256, 256),
        bbox_cfg=dict(
            num_init_random_boxes=9,
            box_jitter_pos=0.1,
            box_jitter_sz=0.5,
            iounet_k=3,
            box_refinement_step_length=2.5e-3,
            box_refinement_iter=10,
            maximal_aspect_ratio=6,
            box_refinement_step_decay=1)),
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
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=8,
    persistent_workers=True,
    test=dict(
        type='GOT10kDataset',
        img_prefix=data_root + 'got10k',
        pipeline=test_pipeline,
        split='test',
        test_mode=True))
