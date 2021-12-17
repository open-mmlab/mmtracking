#### An example of DFF

```python
model = dict(
    type='DFF',  # The name of video detector
    detector=dict(  # Please refer to https://mmdetection.readthedocs.io/en/latest/tutorials/config.html#an-example-of-mask-r-cnn for detailed comments of detector.
        type='FasterRCNN',
        backbone=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(3, ),
            strides=(1, 2, 2, 1),
            dilations=(1, 1, 1, 2),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            style='pytorch',
            init_cfg=dict(
                type='Pretrained', checkpoint='torchvision://resnet50')),
        neck=dict(
            type='ChannelMapper',
            in_channels=[2048],
            out_channels=512,
            kernel_size=3),
        rpn_head=dict(
            type='RPNHead',
            in_channels=512,
            feat_channels=512,
            anchor_generator=dict(
                type='AnchorGenerator',
                scales=[4, 8, 16, 32],
                ratios=[0.5, 1.0, 2.0],
                strides=[16]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[1.0, 1.0, 1.0, 1.0]),
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
                    type='RoIAlign', output_size=7, sampling_ratio=2),
                out_channels=512,
                featmap_strides=[16]),
            bbox_head=dict(
                type='Shared2FCBBoxHead',
                in_channels=512,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=30,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.2, 0.2, 0.2, 0.2]),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0))),
        train_cfg=dict(
            rpn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False),
                allowed_border=0,
                pos_weight=-1,
                debug=False),
            rpn_proposal=dict(
                nms_across_levels=False,
                nms_pre=6000,
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
                nms_pre=6000,
                nms_post=300,
                max_num=300,
                nms_thr=0.7,
                min_bbox_size=0),
            rcnn=dict(
                score_thr=0.0001,
                nms=dict(type='nms', iou_threshold=0.5),
                max_per_img=100))),
    motion=dict(
        type='FlowNetSimple',  # The name of motion model
        img_scale_factor=0.5, # the scale factor to downsample/upsample the input image of motion model
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://download.openmmlab.com/mmtracking/pretrained_weights/flownet_simple.pth'  # noqa: E501
        )), # The pretrained weights of FlowNetSimple
    train_cfg=None,
    test_cfg=dict(key_frame_interval=10))  # The interval of key frame during testing
dataset_type = 'ImagenetVIDDataset'  # Dataset type, this will be used to define the dataset
data_root = 'data/ILSVRC/'  # Root path of data
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],  # Mean values used to pre-training the pre-trained backbone models
    std=[58.395, 57.12, 57.375],  # Standard variance used to pre-training the pre-trained backbone models
    to_rgb=True)  # The channel orders of image used to pre-training the pre-trained backbone models
train_pipeline = [  # Training pipeline
    dict(type='LoadMultiImagesFromFile'),  # First pipeline to load multi images from files path
    dict(
        type='SeqLoadAnnotations',  # Second pipeline to load annotations for multi images
        with_bbox=True,  # Whether to use bounding box, True for detection
        with_track=True),  # Whether to use instance ids, True for detection
    dict(type='SeqResize',   # Augmentation pipeline that resize the multi images and their annotations
        img_scale=(1000, 600),  # The largest scale of image
        keep_ratio=True),  # whether to keep the ratio between height and width.
    dict(
        type='SeqRandomFlip',  # Augmentation pipeline that flip the multi images and their annotations
        share_params=True,
        flip_ratio=0.5),  # The ratio or probability to flip
    dict(
        type='SeqNormalize',  # Augmentation pipeline that normalize the input multi images
        mean=[123.675, 116.28, 103.53],  # These keys are the same of img_norm_cfg since the
        std=[58.395, 57.12, 57.375],  # keys of img_norm_cfg are used here as arguments
        to_rgb=True),
    dict(type='SeqPad',  # Padding config
        size_divisor=16),  # The number the padded images should be divisible
    dict(
        type='VideoCollect',  # Pipeline that decides which keys in the data should be passed to the video detector
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_instance_ids']),
    dict(type='ConcatVideoReferences'),  # Pipeline that concats references images
    dict(type='SeqDefaultFormatBundle',  # Default format bundle to gather data in the pipeline
        ref_prefix='ref')  # The prefix key for reference images.
]
test_pipeline = [
    dict(type='LoadImageFromFile'),  # First pipeline to load images from file path
    dict(
        type='MultiScaleFlipAug',  # An encapsulation that encapsulates the testing augmentations
        img_scale=(1000, 600),  # Decides the largest scale for testing, used for the Resize pipeline
        flip=False,  # Whether to flip images during testing
        transforms=[
            dict(type='Resize',  # Use resize augmentation
                keep_ratio=True),  # Whether to keep the ratio between height and width, the img_scale set here will be suppressed by the img_scale set above.
            dict(type='RandomFlip'),  # Thought RandomFlip is added in pipeline, it is not used because flip=False
            dict(
                type='Normalize',  # Normalization config, the values are from img_norm_cfg
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=16), # Padding config to pad images divisible by 16.
            dict(type='ImageToTensor', keys=['img']),  # convert image to tensor
            dict(type='VideoCollect', keys=['img'])  # Collect pipeline that collect necessary keys for testing.
        ])
]
data = dict(
    samples_per_gpu=1,  # Batch size of a single GPU
    workers_per_gpu=2,  # Worker to pre-fetch data for each single GPU
    train=[
        dict(  # Train dataset config
            type='ImagenetVIDDataset',  # Type of dataset
            ann_file='data/ILSVRCannotations/imagenet_vid_train.json',  # Path of annotation file
            img_prefix='data/ILSVRCData/VID',  # Prefix of image path
            ref_img_sampler=dict(  # configuration for sampling reference images
                num_ref_imgs=1,
                frame_range=9,
                filter_key_img=False,
                method='uniform'),
            pipeline=train_pipeline),  # pipeline, this is passed by the train_pipeline created before.
        dict(
            type='ImagenetVIDDataset',
            load_as_video=False,
            ann_file='data/ILSVRCannotations/imagenet_det_30plus1cls.json',
            img_prefix='data/ILSVRCData/DET',
            ref_img_sampler=dict(
                num_ref_imgs=1,
                frame_range=0,
                filter_key_img=False,
                method='uniform'),
            pipeline=train_pipeline)
    ],
    val=dict(  # Validation dataset config
        type='ImagenetVIDDataset',
        ann_file='data/ILSVRCannotations/imagenet_vid_val.json',
        img_prefix='data/ILSVRCData/VID',
        ref_img_sampler=None,
        pipeline=test_pipeline,  # Pipeline is passed by test_pipeline created before
        test_mode=True),
    test=dict(  # Test dataset config, modify the ann_file for test-dev/test submission
        type='ImagenetVIDDataset',
        ann_file='data/ILSVRCannotations/imagenet_vid_val.json',
        img_prefix='data/ILSVRCData/VID',
        ref_img_sampler=None,
        pipeline=test_pipeline,  # Pipeline is passed by test_pipeline created before
        test_mode=True))
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)  # Config used to build optimizer, support all the optimizers in PyTorch whose arguments are also the same as those in PyTorch
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))  # Config used to build the optimizer hook, refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/optimizer.py#L8 for implementation details.
checkpoint_config = dict(interval=1)  # Config to set the checkpoint hook, Refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py for implementation.
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])  # config to register logger hook
dist_params = dict(backend='nccl', port='29500') # Parameters to setup distributed training, the port is set to 29500 by default
log_level = 'INFO'  # The level of logging.
load_from = None  # load models as a pre-trained model from a given path. This will not resume training.
resume_from = None  # Resume checkpoints from a given path, the training will be resumed from the epoch when the checkpoint's is saved.
workflow = [('train', 1)]  # Workflow for runner. [('train', 1)] means there is only one workflow and the workflow named 'train' is executed once. The workflow trains the model by 7 epochs according to the total_epochs.
lr_config = dict(  # Learning rate scheduler config used to register LrUpdater hook
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
    step=[2, 5])
total_epochs = 7  # Total epochs to train the model
evaluation = dict(metric=['bbox'], interval=7)  # The config to build the evaluation hook
work_dir = '../mmtrack_output/tmp'  # Directory to save the model checkpoints and logs for the current experiments.
```
