# model settings
_base_ = [
    './qdtrack_faster-rcnn_r50_fpn_4e_base.py', '../../_base_/datasets/tao.py'
]
model = dict(
    type='QDTrack',
    data_preprocessor=dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    detector=dict(
        backbone=dict(
            depth=101,
            init_cfg=dict(
                type='Pretrained', checkpoint='torchvision://resnet101')),
        roi_head=dict(
            bbox_head=dict(
                loss_bbox=dict(type='L1Loss', loss_weight=1.0),
                num_classes=482)),
        test_cfg=dict(
            rcnn=dict(
                score_thr=0.0001,
                nms=dict(type='nms', iou_threshold=0.5),
                max_per_img=300)),
        init_cfg=None),
    tracker=dict(
        _delete_=True,
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
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 1000,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='mmdet.MultiStepLR',
        begin=0,
        end=24,
        by_epoch=True,
        milestones=[16, 22])
]
# runtime settings
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=24, val_begin=16, val_interval=2)

# evaluator
val_evaluator = dict(type='CocoVideoMetric', metric=['bbox'])
test_evaluator = val_evaluator
