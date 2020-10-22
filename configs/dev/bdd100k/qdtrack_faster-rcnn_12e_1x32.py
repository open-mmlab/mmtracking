_base_ = [
    '../../_base_/models/faster_rcnn_r50_fpn.py',
    '../../_base_/datasets/bdd100k_track_joint.py',
    '../../_base_/default_runtime.py'
]
model = dict(
    type='QDTrack',
    pretrains=None,
    frozen_modules=None,
    detector=dict(roi_head=dict(bbox_head=dict(num_classes=8))),
    track_head=dict(
        type='QuasiDenseTrackHead',
        multi_positive=True,
        roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        roi_assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.5,
            match_low_quality=False,
            ignore_iof_thr=-1),
        roi_sampler=dict(
            type='CombinedSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=3,
            add_gt_as_proposals=True,
            pos_sampler=dict(type='InstanceBalancedPosSampler'),
            neg_sampler=dict(
                type='IoUBalancedNegSampler',
                floor_thr=-1,
                floor_fraction=0,
                num_bins=3)),
        embed_head=dict(
            type='QuasiDenseEmbedHead',
            num_convs=4,
            num_fcs=1,
            embed_channels=256,
            norm_cfg=dict(type='GN', num_groups=32),
            loss_track=dict(type='MultiPosCrossEntropyLoss', loss_weight=0.25),
            loss_track_aux=dict(
                type='L2Loss',
                neg_pos_ub=3,
                pos_margin=0,
                neg_margin=0.3,
                hard_mining=True,
                loss_weight=1.0))),
    tracker=dict(
        type='QuasiDenseEmbedTracker',
        init_score_thr=0.7,
        obj_score_thr=0.3,
        match_score_thr=0.5,
        memo_tracklet_frames=10,
        memo_backdrop_frames=1,
        memo_momentum=0.8,
        nms_conf_thr=0.5,
        nms_backdrop_iou_thr=0.3,
        nms_class_iou_thr=0.7,
        with_cats=True,
        match_metric='bisoftmax'))
data = dict(samples_per_gpu=1, workers_per_gpu=1)
# optimizer
optimizer = dict(type='SGD', lr=0.04, momentum=0.9, weight_decay=0.0001)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 1000,
    step=[8, 11])
# runtime settings
total_epochs = 12
evaluation = dict(metric=['bbox', 'track'], interval=3)
log_config = dict(interval=50)
search_metrics = ['MOTA', 'IDF1', 'FN', 'FP', 'IDs', 'MT', 'ML']
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
