_base_ = [
    './tracktor_faster-rcnn_r50-fpn_8xb2-4e_mot17halftrain'
    '_test-mot17halfval.py'
]

model = dict(
    tracker=dict(
        type='TracktorTracker',
        obj_score_thr=[0.4, 0.5, 0.6],
        regression=dict(
            obj_score_thr=[0.4, 0.5, 0.6],
            nms=dict(type='nms', iou_threshold=0.6),
            match_iou_thr=[0.3, 0.5]),
        reid=dict(
            num_samples=10,
            img_scale=(256, 128),
            img_norm_cfg=None,
            match_score_thr=2.0,
            match_iou_thr=0.2),
        momentums=None,
        num_frames_retain=10))
