_base_ = [
    '../../_base_/models/faster_rcnn_r50_fpn.py',
    '../../_base_/datasets/mot_challenge.py', '../../_base_/default_runtime.py'
]
model = dict(
    type='DeepSORT',
    detector=dict(
        rpn_head=dict(bbox_coder=dict(clip_border=False)),
        roi_head=dict(
            bbox_head=dict(bbox_coder=dict(clip_border=False), num_classes=1)),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth'  # noqa: E501
        )),
    motion=dict(type='KalmanFilter', center_only=False),
    tracker=dict(
        type='SortTracker', obj_score_thr=0.5, match_iou_thr=0.5, reid=None))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=1.0 / 100,
    step=[3])
# runtime settings
total_epochs = 4
evaluation = dict(metric=['bbox', 'track'], interval=1)
search_metrics = ['MOTA', 'IDF1', 'FN', 'FP', 'IDs', 'MT', 'ML']
