_base_ = [
    '../../_base_/models/faster_rcnn_r50_fpn.py',
    '../../_base_/datasets/mot17.py', '../../_base_/default_runtime.py'
]
model = dict(
    type='Tracktor',
    pretrains=dict(
        detector='ckpts/person_detector/' +
        'faster_rcnn_r50_caffe_fpn_person_ap551.pth',
        reid='ckpts/tracktor/reid/reid_r50_tracktor_iter25245.pth'),
    detector=dict(roi_head=dict(bbox_head=dict(num_classes=1))),
    reid=dict(
        type='BaseReID',
        backbone=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(3, ),
            style='pytorch'),
        neck=dict(type='GlobalAveragePooling', kernel_size=(8, 4), stride=1),
        head=dict(
            type='LinearReIDHead',
            num_fcs=1,
            in_channels=2048,
            fc_channels=1024,
            out_channels=128,
            norm_cfg=dict(type='BN1d'),
            act_cfg=dict(type='ReLU'))),
    motion=[
        dict(
            type='CameraMotionCompensation',
            warp_mode='cv2.MOTION_EUCLIDEAN',
            num_iters=100,
            stop_eps=0.00001),
        dict(type='LinearMotion', num_samples=2, center_motion=True)
    ],
    tracker=dict(
        type='TracktorTracker',
        obj_score_thr=0.5,
        nms_thr=0.3,
        reg_obj_score_thr=0.5,
        reg_nms_thr=0.6,
        reid_sim_thr=2.0,
        reid_iou_thr=0.2,
        num_frames_retain=10))
data = dict(samples_per_gpu=1, workers_per_gpu=1)
# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
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
log_config = dict(interval=50)
search_metrics = ['MOTA', 'IDF1', 'FN', 'FP', 'IDs', 'MT', 'ML']
