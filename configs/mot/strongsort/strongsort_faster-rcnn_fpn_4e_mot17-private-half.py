_base_ = [
    '../../_base_/models/faster_rcnn_r50_fpn.py',
    '../../_base_/datasets/mot_challenge.py', '../../_base_/default_runtime.py'
]

default_hooks = dict(
    # visualization=dict(type='TrackVisualizationHook', draw=True),
)

model = dict(
    type='StrongSORT',
    detector=dict(
        rpn_head=dict(bbox_coder=dict(clip_border=False)),
        roi_head=dict(
            bbox_head=dict(bbox_coder=dict(clip_border=False), num_classes=1)),
        # test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.8)),  # dyh
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth'  # noqa: E501
        )),
    kalman=dict(type='KalmanFilter', center_only=False),
    reid=dict(
        type='BaseReID',
        data_preprocessor=None,
        backbone=dict(
            type='mmcls.ResNet',
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
            num_classes=380,
            loss_cls=dict(type='mmcls.CrossEntropyLoss', loss_weight=1.0),
            loss_triplet=dict(type='TripletLoss', margin=0.3, loss_weight=1.0),
            norm_cfg=dict(type='BN1d'),
            act_cfg=dict(type='ReLU')),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://download.openmmlab.com/mmtracking/mot/reid/tracktor_reid_r50_iter25245-a452f51f.pth'  # noqa: E501
            # 'https://download.openmmlab.com/mmtracking/mot/reid/reid_r50_6e_mot17-4bf6b63d.pth'  # noqa: E501
        )),
    tracker=dict(
        type='StrongSORTTracker',
        obj_score_thr=0.5,
        # obj_score_thr=0.6,
        reid=dict(
            # num_samples=10,
            num_samples=None,
            img_scale=(256, 128),
            img_norm_cfg=None,
            match_score_thr=2.0,
            # match_score_thr=0.4
            motion_weight=0,
            # motion_weight=0.02,
        ),
        match_iou_thr=0.5,
        # match_iou_thr=0.7,
        # momentums=None,
        momentums=dict(
            embeds=0.1,
        ),
        num_tentatives=2,
        # num_frames_retain=100
        num_frames_retain=30  # 30
))

train_dataloader = None

train_cfg = None
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
