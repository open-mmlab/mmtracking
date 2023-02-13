_base_ = [
    '../../_base_/models/faster-rcnn_r50_fpn.py',
    '../../_base_/datasets/mot_challenge.py', '../../_base_/default_runtime.py'
]

model = dict(
    type='OCSORT',
    data_preprocessor=dict(
        type='TrackDataPreprocessor',
        pad_size_divisor=32,
        batch_augments=[
            dict(
                type='mmdet.BatchSyncRandomResize',
                random_size_range=(576, 1024),
                size_divisor=32,
                interval=10)
        ]),
    detector=dict(
        rpn_head=dict(bbox_coder=dict(clip_border=False)),
        roi_head=dict(
            bbox_head=dict(bbox_coder=dict(clip_border=False), num_classes=1)),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth'  # noqa: E501
        )),
    motion=dict(type='KalmanFilter'),
    tracker=dict(
        type='OCSORTTracker',
        obj_score_thr=0.3,
        init_track_thr=0.7,
        weight_iou_with_det_scores=True,
        match_iou_thr=0.3,
        num_tentatives=3,
        vel_consist_weight=0.2,
        vel_delta_t=3,
        num_frames_retain=30))

train_cfg = None
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
