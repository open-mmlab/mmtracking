#### An example of Tracktor

```python
model = dict(
    type='Tracktor',  # The name of the multiple object tracker
    detector=dict(  # Please refer to https://github.com/open-mmlab/mmdetection/blob/master/docs/tutorials/config.md#an-example-of-mask-r-cnn for detailed comments of detector.
        type='FasterRCNN',
        backbone=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            style='pytorch',
            init_cfg=dict(
                type='Pretrained', checkpoint='torchvision://resnet50')),
        neck=dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=5),
        rpn_head=dict(
            type='RPNHead',
            in_channels=256,
            feat_channels=256,
            anchor_generator=dict(
                type='AnchorGenerator',
                scales=[8],
                ratios=[0.5, 1.0, 2.0],
                strides=[4, 8, 16, 32, 64]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[1.0, 1.0, 1.0, 1.0],
                clip_border=False),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            loss_bbox=dict(
                type='SmoothL1Loss', beta=0.1111111111111111,
                loss_weight=1.0)),
        roi_head=dict(
            type='StandardRoIHead',
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]),
            bbox_head=dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2],
                    clip_border=False),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', loss_weight=1.0))),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth'  # noqa: E501
        ), # The pretrained weights of detector. It may also used for testing.
        train_cfg=dict(
            rpn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    match_low_quality=True,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False),
                allowed_border=-1,
                pos_weight=-1,
                debug=False),
            rpn_proposal=dict(
                nms_across_levels=False,
                nms_pre=2000,
                nms_post=1000,
                max_num=1000,
                nms_thr=0.7,
                min_bbox_size=0),
            rcnn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)),
        test_cfg=dict(
            rpn=dict(
                nms_across_levels=False,
                nms_pre=1000,
                nms_post=1000,
                max_num=1000,
                nms_thr=0.7,
                min_bbox_size=0),
            rcnn=dict(
                score_thr=0.05,
                nms=dict(type='nms', iou_threshold=0.5),
                max_per_img=100))),
    reid=dict(  # The config of the ReID model
        type='BaseReID',  # The name of the motion model
        backbone=dict(  # The config of the backbone of the ReID model
            type='ResNet', # The type of the backbone, refer to https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/backbones/resnet.py#L288 for more details.
            depth=50,  # The depth of backbone, usually it is 50 or 101 for ResNet and ResNext backbones.
            num_stages=4,  # Number of stages of the backbone.
            out_indices=(3, ),  # The index of output feature maps produced in each stages
            style='pytorch'),  # The style of backbone, 'pytorch' means that stride 2 layers are in 3x3 conv, 'caffe' means stride 2 layers are in 1x1 convs.
        neck=dict(type='GlobalAveragePooling', kernel_size=(8, 4), stride=1),  # The config of the neck of the ReID model. Generally it is a global average pooling module.
        head=dict(  # The config of the head of the ReID model.
            type='LinearReIDHead',  # The type of the classification head
            num_fcs=1,  # Number of the fully-connected layers in the head
            in_channels=2048,  # The number of the input channels
            fc_channels=1024,  # The number of channels of fc layers
            out_channels=128,  # The number of the output channels
            norm_cfg=dict(type='BN1d'),  # The config of the normalization modules
            act_cfg=dict(type='ReLU')),  # The config of the activation modules
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://download.openmmlab.com/mmtracking/mot/reid/reid_r50_6e_mot17-4bf6b63d.pth'  # noqa: E501
        )), # The pretrained weights of reid model. It may also used for testing.
    motion=dict(  # The config of the motion model
        type='CameraMotionCompensation',  # The name of the motion model
        warp_mode='cv2.MOTION_EUCLIDEAN',  # The warping mode
        num_iters=100, # The number of the iterations
        stop_eps=1e-05),  # The threshold of termination
    tracker=dict(  # The config of the tracker
        type='TracktorTracker',  # The name of the tracker
        obj_score_thr=0.5,  # The score threshold to filter the detected objects
        regression=dict(  # The config of the regression part in Tracktor
            obj_score_thr=0.5,  # The score threshold to filter the regressed objects
            nms=dict(type='nms', iou_threshold=0.6),  # The nms config to filter the regressed objects
            match_iou_thr=0.3),  # The IoU threshold to filter the detected objects
        reid=dict(  # The config about the testing process of the ReID model
            num_samples=10,  # The maximum number of samples to calculate the feature embeddings
            img_scale=(256, 128),  # The input scale of the ReID model
            img_norm_cfg=None,  # The normalization config of the input of the ReID model. None means consistent with the backbone
            match_score_thr=2.0,  # The threshold for feature similarity
            match_iou_thr=0.2),  # The threshold for IoU matching
        momentums=None,  # The momentums to update the buffers
        num_frames_retain=10))  # The maximum number of frames to retain disappeared tracks
# The configs below are consistent with video object tracking. Please refer to `config_vid.md` for details.
dataset_type = 'MOTChallengeDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadMultiImagesFromFile', to_float32=True),
    dict(type='SeqLoadAnnotations', with_bbox=True, with_track=True),
    dict(
        type='SeqResize',
        img_scale=(1088, 1088),
        share_params=True,
        ratio_range=(0.8, 1.2),
        keep_ratio=True,
        bbox_clip_border=False),
    dict(type='SeqPhotoMetricDistortion', share_params=True),
    dict(
        type='SeqRandomCrop',
        share_params=False,
        crop_size=(1088, 1088),
        bbox_clip_border=False),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
    dict(
        type='SeqNormalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='SeqPad', size_divisor=32),
    dict(type='MatchInstances', skip_nomatch=True),
    dict(
        type='VideoCollect',
        keys=[
            'img', 'gt_bboxes', 'gt_labels', 'gt_match_indices',
            'gt_instance_ids'
        ]),
    dict(type='SeqDefaultFormatBundle', ref_prefix='ref')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1088, 1088),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='VideoCollect', keys=['img'])
        ])
]
data_root = 'data/MOT17/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='MOTChallengeDataset',
        visibility_thr=-1,
        ann_file='data/MOT17/annotations/train_cocoformat.json',
        img_prefix='data/MOT17/train',
        ref_img_sampler=dict(
            num_ref_imgs=1,
            frame_range=10,
            filter_key_img=True,
            method='uniform'),
        pipeline=train_pipeline),
    val=dict(
        type='MOTChallengeDataset',
        ann_file='data/MOT17/annotations/train_cocoformat.json',
        img_prefix='data/MOT17/train',
        ref_img_sampler=None,
        pipeline=test_pipeline),
    test=dict(
        type='MOTChallengeDataset',
        ann_file='data/MOT17/annotations/train_cocoformat.json',
        img_prefix='data/MOT17/train',
        ref_img_sampler=None,
        pipeline=test_pipeline))
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl', port='29500')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.01,
    step=[3])
total_epochs = 4
evaluation = dict(metric=['bbox', 'track'], interval=1)
search_metrics = ['MOTA', 'IDF1', 'FN', 'FP', 'IDs', 'MT', 'ML']
test_set = 'train'
work_dir = None
```
