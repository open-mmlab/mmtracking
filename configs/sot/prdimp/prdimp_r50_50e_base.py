_base_ = ['../../_base_/default_runtime.py']

randomness = dict(seed=1, deterministic=False)

# model setting
model = dict(
    type='Prdimp',
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
        type='PrdimpClsHead',
        in_dim=1024,
        out_dim=512,
        filter_initializer=dict(
            type='FilterInitializer',
            filter_size=4,
            feature_dim=512,
            feature_stride=16),
        filter_optimizer=dict(
            type='PrdimpFilterOptimizer',
            num_iters=5,
            feat_stride=16,
            init_step_length=1.0,
            init_filter_regular=0.05,
            gauss_sigma=0.9,
            alpha_eps=0.05,
            min_filter_regular=0.05,
            label_thres=0),
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
            init_update_iters=10, update_iters=2, hard_neg_iters=1)),
    bbox_head=dict(
        type='IouNetHead',
        in_dim=(4 * 128, 4 * 256),
        pred_in_dim=(256, 256),
        pred_inter_dim=(256, 256),
        bbox_cfg=dict(
            num_init_random_boxes=9,
            box_jitter_pos=0.1,
            box_jitter_sz=0.5,
            iounet_topk=3,
            box_refine_step_length=2.5e-3,
            box_refine_iter=10,
            max_aspect_ratio=6,
            box_refine_step_decay=1)),
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
