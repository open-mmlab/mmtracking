_base_ = [
    '../../_base_/models/yolov7_l_syncbn_fast.py',
    '../../_base_/datasets/mot_challenge.py', 
    '../../_base_/default_runtime.py'
]
model = dict(
    type='DeepSORT',
    detector=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://download.openmmlab.com/mmyolo/v0/yolov7/yolov7_l_syncbn_fast_8x16b-300e_coco/yolov7_l_syncbn_fast_8x16b-300e_coco_20221123_023601-8113c0eb.pth'  # noqa: E501
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
    tracker=dict(
        type='SORTTracker',
        obj_score_thr=0.5,
        reid=dict(
            num_samples=10,
            img_scale=(256, 128),
            img_norm_cfg=None,
            match_score_thr=2.0),
        match_iou_thr=0.5,
        momentums=None,
        num_tentatives=2,
        num_frames_retain=100))

train_dataloader = None

train_cfg = None
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
