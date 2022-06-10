# model settings
_base_ = [
    '../../_base_/models/faster_rcnn_r50_fpn.py',
    '../../_base_/datasets/tao.py', '../../_base_/default_runtime.py'
]
model = dict(
    type='QDTrack',
    detector=dict(
        backbone=dict(
            depth=101,
            init_cfg=dict(
                type='Pretrained', checkpoint='../resnet101-63fe2227.pth')),
        roi_head=dict(
            bbox_head=dict(
                loss_bbox=dict(type='L1Loss', loss_weight=1.0),
                num_classes=482)),
        test_cfg=dict(
            rcnn=dict(
                score_thr=0.0001,
                nms=dict(type='nms', iou_threshold=0.5),
                max_per_img=300))),
    track_head=dict(
        type='QuasiDenseTrackHead',
        roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
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
                neg_margin=0.1,
                hard_mining=True,
                loss_weight=1.0)),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0),
        train_cfg=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='CombinedSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=3,
                add_gt_as_proposals=True,
                pos_sampler=dict(type='InstanceBalancedPosSampler'),
                neg_sampler=dict(type='RandomSampler')))),
    tracker=dict(
        type='QuasiDenseTAOTracker',
        init_score_thr=0.0001,
        obj_score_thr=0.0001,
        match_score_thr=0.5,
        memo_frames=10,
        memo_momentum=0.8,
        momentum_obj_score=0.5,
        obj_score_diff_thr=1.0,
        distractor_nms_thr=0.3,
        distractor_score_thr=0.5,
        match_metric='bisoftmax',
        match_with_cosine=True))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 1000,
    step=[16, 22])
total_epochs = 24
load_from = None
resume_from = None
evaluation = dict(metric=['bbox'], start=16, interval=2)
work_dir = None
# TODO: del this
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook', init_kwargs=dict(project='QDtrack'))
    ])
