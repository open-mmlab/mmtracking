_base_ = [
    '../../_base_/models/faster-rcnn_r50_fpn.py',
    '../../_base_/datasets/mot_challenge.py', '../../_base_/default_runtime.py'
]

custom_imports = dict(
    imports=[
        'mmtrack.models.mot.my_deep_sort',
        'mmtrack.models.trackers.my_sort_tracker'
    ],
    allow_failed_imports=False)

model = dict(
    type='MyDeepSORT',
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
        )),
    pose=dict(
        type='TopdownPoseEstimator',
        _scope_='mmpose',
        data_preprocessor=None,
        backbone=dict(
            type='ResNet',
            depth=50,
            init_cfg=dict(
                type='Pretrained', checkpoint='torchvision://resnet50'),
        ),
        head=dict(
            type='HeatmapHead',
            in_channels=2048,
            out_channels=17,
            loss=dict(type='KeypointMSELoss', use_target_weight=True),
            decoder=dict(
                type='MSRAHeatmap',
                input_size=(192, 256),
                heatmap_size=(48, 64),
                sigma=2)),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_256x192-ec54d7f3_20200709.pth'
        ),
        test_cfg=dict(
            flip_test=False,
            flip_mode='heatmap',
            shift_heatmap=True,
        )),
    tracker=dict(
        type='MySORTTracker',
        obj_score_thr=0.5,
        reid=dict(
            num_samples=10,
            img_scale=(256, 128),
            img_norm_cfg=None,
            match_score_thr=2.0),
        match_iou_thr=0.5,
        momentums=None,
        num_tentatives=2,
        num_frames_retain=100,
        pose=True))

train_dataloader = None

train_cfg = None
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
