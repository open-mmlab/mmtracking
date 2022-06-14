# model settings
_base_ = [
    './qdtrack_faster-rcnn_r50_fpn_4e_base.py', '../../_base_/datasets/tao.py'
]
model = dict(
    type='QDTrack',
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
