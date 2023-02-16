_base_ = [
    '../../_base_/models/yolov7_l_syncbn_fast.py',
    '../../_base_/default_runtime.py',
    '../../_base_/datasets/mot_challenge.py'
]

img_scale = (800, 1040)
strides = [8, 16, 32]

model = dict(
    type='QDTrackPlus',
    data_preprocessor=dict(
        _delete_=True,
        type='TrackDataPreprocessor',
        mean=[0, 0, 0],
        std=[255.0, 255.0, 255.0],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    detector=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmyolo/v0/yolov7/yolov7_l_syncbn_fast_8x16b-300e_coco/yolov7_l_syncbn_fast_8x16b-300e_coco_20221123_023601-8113c0eb.pth')  # noqa: E501
        ),
    track_head=dict(
        type='QuasiDenseTrackHead',
        roi_extractor=dict(
            _scope_='mmdet',
            type='GenericRoIExtractor',
            aggregation='concat',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=1792, # 256 + 512 + 1024
            featmap_strides=strides),
        embed_head=dict(
            type='QuasiDenseEmbedHead',
            num_convs=4,
            num_fcs=1,
            in_channels=1792,
            embed_channels=256,
            norm_cfg=dict(type='GN', num_groups=32),
            loss_track=dict(type='MultiPosCrossEntropyLoss', loss_weight=0.25),
            loss_track_aux=dict(
                type='L2Loss',
                neg_pos_ub=3,
                pos_margin=0,
                neg_margin=0.1,
                hard_mining=True,
                loss_weight=1.0)),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0),
        train_cfg=dict(
            assigner=dict(
                _scope_='mmdet',
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                _scope_='mmdet',
                type='CombinedSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=3,
                add_gt_as_proposals=True,
                pos_sampler=dict(type='InstanceBalancedPosSampler'),
                neg_sampler=dict(type='RandomSampler')))),
    tracker=dict(
        type='QuasiDenseTracker',
        init_score_thr=0.9,
        obj_score_thr=0.5,
        match_score_thr=0.5,
        memo_tracklet_frames=30,
        memo_backdrop_frames=1,
        memo_momentum=0.8,
        nms_conf_thr=0.5,
        nms_backdrop_iou_thr=0.3,
        nms_class_iou_thr=0.7,
        with_cats=True,
        match_metric='bisoftmax'),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0)))

# optimizer
lr = 0.001
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=lr, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2))

# some hyper parameters
# training settings
total_epochs = 80
num_last_epochs = 10
resume_from = None
interval = 5

# learning policy
param_scheduler = [
    dict(
        # use quadratic formula to warm up 1 epochs
        # and lr is updated by iteration
        # TODO: fix default scope in get function
        type='mmdet.QuadraticWarmupLR',
        by_epoch=True,
        begin=0,
        end=1,
        convert_to_iter_based=True),
    dict(
        # use cosine lr from 1 to 70 epoch
        type='mmdet.CosineAnnealingLR',
        eta_min=lr * 0.05,
        begin=1,
        T_max=total_epochs - num_last_epochs,
        end=total_epochs - num_last_epochs,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        # use fixed lr during last 10 epochs
        type='mmdet.ConstantLR',
        by_epoch=True,
        factor=1,
        begin=total_epochs - num_last_epochs,
        end=total_epochs,
    )
]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=total_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

custom_hooks = [
    dict(type='mmdet.SyncNormHook', priority=48),
    dict(
        type='mmdet.EMAHook',
        ema_type='mmdet.ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        priority=49)
]
default_hooks = dict(checkpoint=dict(interval=1))