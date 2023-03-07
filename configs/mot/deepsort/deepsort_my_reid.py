_base_ = [
    '../../_base_/models/faster-rcnn_r50_fpn.py',
    '../../_base_/datasets/mot_challenge.py', '../../_base_/default_runtime.py'
]

custom_imports = dict(
    imports=['mmtrack.models.reid.my_reid'], allow_failed_imports=False)

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
    reid=dict(
        type='MyReID',
        model_name='osnet_x1_0',
        model_path='...',
        device='cuda',
    ),
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
        num_frames_retain=100,
        biou=True))

train_dataloader = None

train_cfg = None
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# dataset settings
dataset_type = 'MOTChallengeDataset'
data_root = '../../datasets/AIC23_Track1_MTMC_Tracking/'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadTrackAnnotations', with_instance_id=True),
    dict(type='mmdet.Resize', scale=(1088, 1088), keep_ratio=True),
    dict(type='PackTrackInputs', pack_single_img=True)
]

# dataloader
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='VideoSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/validation_cocoformat_subset_0.2_consec.json',
        data_prefix=dict(img_path='validation'),
        metainfo=dict(CLASSES=('person', )),
        ref_img_sampler=None,
        load_as_video=True,
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader