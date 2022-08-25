#### An example of SiameseRPN++

```python
cudnn_benchmark = True  ## True for accelerating the training speed
crop_size = 511  # Crop size around a bounding box
exemplar_size = 127  # Exemplar size
search_size = 255  #  Search size

# model settings
model = dict(
    type='SiamRPN',  # the type of single object tracker
    backbone=dict(  # The config of backbone
        type='SOTResNet',  # The type of the backbone
        depth=50,  # The depth of backbone, usually it is 50 for ResNet backbones.
        out_indices=(1, 2, 3),  # The index of output feature maps produced in each stage
        frozen_stages=4,  # The weights in the first 1 stage are fronzen
        strides=(1, 2, 1, 1),  # The stride of conv in each stage
        dilations=(1, 1, 2, 4),  # The dilation of conv in each stage
        norm_eval=True, # Whether to freeze the statistics in BN
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://download.openmmlab.com/mmtracking/pretrained_weights/sot_resnet50.model'  # noqa: E501
        )), # The pretrained weights of backbone
    neck=dict(
        type='ChannelMapper',  # The neck is ChannelMapper.
        in_channels=[512, 1024, 2048],  # The input channels, this is consistent with the output channels of backbone
        out_channels=256,  # The output channels of each level of the pyramid feature map
        kernel_size=1,  # kernel size of convs in ChannelMapper
        norm_cfg=dict(type='BN'),  # The config of normalization layers.
        act_cfg=None),  # The config of activation layers.
    head=dict(
        type='SiameseRPNHead',  # The type of head.
        anchor_generator=dict(  # The config of anchor generator
            type='SiameseRPNAnchorGenerator',  # anchor generator for SiameseRPN++
            strides=[8],  # The strides of the anchor generator. This is consistent with the FPN feature strides. The strides will be taken as base_sizes if base_sizes is not set.
            ratios=[0.33, 0.5, 1, 2, 3],  # The ratio between height and width.
            scales=[8]),  # Basic scale of the anchor, the area of the anchor in one position of a feature map will be scale * base_sizes
        in_channels=[256, 256, 256],  # The input channels, this is consistent with the output channels of neck
        weighted_sum=True,  # If True, use learnable weights to weightedly sum the output of multi heads in siamese rpn , otherwise, use averaging.
        bbox_coder=dict(  # Config of box coder to encode and decode the boxes during training and testing
            type='DeltaXYWHBBoxCoder',  # Type of box coder. 'DeltaXYWHBBoxCoder' is applied for most of methods. Refer to https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/coder/delta_xywh_bbox_coder.py#L9 for more details.
            target_means=[0., 0., 0., 0.],  # The target means used to encode and decode boxes
            target_stds=[1., 1., 1., 1.]),  # The standard variance used to encode and decode boxes
        loss_cls=dict(  # Config of loss function for the classification branch
            type='CrossEntropyLoss', # Type of loss for classification branch, we also support FocalLoss etc.
            reduction='sum',
            loss_weight=1.0),  # Loss weight of the classification branch.
        loss_bbox=dict(  # Config of loss function for the regression branch.
            type='L1Loss',  # Type of loss, we also support many IoU Losses and smooth L1-loss, etc. Refer to https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/losses/smooth_l1_loss.py#L56 for implementation.
            reduction='sum',
            loss_weight=1.2)),  # Loss weight of the regression branch.
    train_cfg=dict(  # Config of training hyperparameters for rpn and rcnn
        rpn=dict(  # Training config of rpn
            assigner=dict(  # Config of assigner
                type='MaxIoUAssigner',  # Type of assigner, MaxIoUAssigner is used for many common detectors. Refer to https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/assigners/max_iou_assigner.py#L10 for more details.
                pos_iou_thr=0.6,  # IoU >= threshold 0.6 will be taken as positive samples
                neg_iou_thr=0.3,  # IoU < threshold 0.3 will be taken as negative samples
                min_pos_iou=0.6,  # The minimal IoU threshold to take boxes as positive samples
                match_low_quality=False),  # Whether to match the boxes under low quality (see API doc for more details).
            sampler=dict(  # Config of positive/negative sampler
                type='RandomSampler',  # Type of sampler, PseudoSampler and other samplers are also supported. Refer to https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/samplers/random_sampler.py#L8 for implementation details.
                num=64,  # Number of samples when the examplar image and search image are positive pair.
                pos_fraction=0.25,  # The ratio of positive samples in the total samples when the examplar image and search image are positive pair.
                add_gt_as_proposals=False),  # Whether add GT as proposals after sampling.
            num_neg=16,  # Number of negative samples when the examplar image and search image are negative pair.
            exemplar_size=exemplar_size,
            search_size=search_size)),
    test_cfg=dict(
        exemplar_size=exemplar_size,
        search_size=search_size,
        context_amount=0.5,  # the amount of context
        center_size=7,  # the size of cropping the center feature maps of examplar image
        rpn=dict(penalty_k=0.05, window_influence=0.42, lr=0.38)))  # used to smooth the predicted tracking box.

data_root = 'data/'  # Dataset type, this will be used to define the dataset
train_pipeline = [
    dict(
        type='PairSampling',  # sampling method in training
        frame_range=5,  # the sampling range of search frames in the same video for template frame
        pos_prob=0.8,  # the probility of sampling positive sample pairs
        filter_template_img=False), # whether to exclude template frame when sampling search frame
    dict(type='LoadMultiImagesFromFile',  # First pipeline to load multi images from files path
        to_float32=True),  # convert the image to np.float32
    dict(type='SeqLoadAnnotations',  # Second pipeline to load annotations for multi images
        with_bbox=True),  # Whether to use bounding box, True for detection
    dict(
        type='SeqCropLikeSiamFC',  # crop images like SiamFC does.
        context_amount=0.5,  # the amount of context
        exemplar_size=exemplar_size,
        crop_size=crop_size),
    dict(
        type='SeqShiftScaleAug',  #  shift and rescale images
        target_size=[exemplar_size, search_size],  # the target size of each image
        shift=[4, 64],  # the max shift offset for each image
        scale=[0.05, 0.18]),  # the max rescale offset for each image
    dict(type='SeqColorAug',  # color augmentation
        prob=[1.0, 1.0]),  # the probability color augmentaion for each image
    dict(type='SeqBlurAug',  # blur augmentation
        prob=[0.0, 0.2]),  # the probability color augmentaion for each image
    dict(type='VideoCollect', keys=['img', 'gt_bboxes', 'is_positive_pairs']),  # Pipeline that decides which keys in the data should be passed to the video detector
    dict(type='ConcatVideoReferences'),  # Pipeline that concats references images
    dict(type='SeqDefaultFormatBundle',  # Default format bundle to gather data in the pipeline
        ref_prefix='search')  # The prefix key for reference images.
]
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),  # First pipeline to load images from file path
    dict(type='LoadAnnotations', with_bbox=True), # Second pipeline to load annotations for image
    dict(
        type='MultiScaleFlipAug',  # An encapsulation that encapsulates the testing augmentations
        scale_factor=1,  # the scale factor to resize the input image
        flip=False,  # Whether to flip images during testing
        transforms=[
            dict(type='VideoCollect', keys=['img', 'gt_bboxes']),  # Collect pipeline that collect necessary keys for testing.
            dict(type='ImageToTensor', keys=['img'])  # convert image to tensor
        ])
]
# dataset settings
data = dict(
    samples_per_gpu=28,  # Batch size of a single GPU
    workers_per_gpu=8,  # Worker to pre-fetch data for each single GPU
    persistent_workers=True,  # Setting of persistent workers
    samples_per_epoch=600000,  # The number of training samples per epoch
    train=dict(
        type='RandomSampleConcatDataset',  # Sampling the comcatenated datasets ramdomly
        dataset_sampling_weights=[0.25, 0.2, 0.55],  # The sampling weights of concatenated dataset
        dataset_cfgs=[
            dict(
                type='SOTImageNetVIDDataset',  # Type of dataset
                ann_file=data_root +
                'ILSVRC/annotations/imagenet_vid_train.json',  # Path of annotation file
                img_prefix=data_root + 'ILSVRC/Data/VID',  # Prefix of image path
                pipeline=train_pipeline,  # pipeline, this is passed by the train_pipeline created before.
                split='train',  # Split of dataset
                test_mode=False),  # Whether test mode
            dict(
                type='SOTCocoDataset',  # Type of dataset
                ann_file=data_root +
                'coco/annotations/instances_train2017.json',  # Path of annotation file
                img_prefix=data_root + 'coco/train2017',  # Prefix of image path
                pipeline=train_pipeline,  # pipeline, this is passed by the train_pipeline created before.
                split='train',  # Split of dataset
                test_mode=False),  # Whether test mode
            dict(
                type='SOTCocoDataset',  # Type of dataset
                ann_file=data_root +
                'ILSVRC/annotations/imagenet_det_30plus1cls.json',  # Path of annotation file
                img_prefix=data_root + 'ILSVRC/Data/DET',  # Prefix of image path
                pipeline=train_pipeline,  # pipeline, this is passed by the train_pipeline created before.
                split='train',  # Split of dataset
                test_mode=False)  # Whether test mode
        ]),
    val=dict(  # Validation dataset config
        type='LaSOTDataset',
        ann_file=data_root + 'lasot/annotations/lasot_test_infos.txt',  # Path of dataset information file
        img_prefix=data_root + 'lasot/LaSOTBenchmark',  # Prefix of image path
        pipeline=test_pipeline,  # Pipeline is passed by test_pipeline created before
        split='test',   # split of dataset
        test_mode=True,  # whether test mode
        only_eval_visible=True),  # whether to only evaluate the method on frames where the object is visible in LaSOT
    test=dict(  # Test dataset config, modify the ann_file for test-dev/test submission
        type='LaSOTDataset',
        ann_file=data_root + 'lasot/annotations/lasot_test_infos.txt',  # Path of dataset information file
        img_prefix=data_root + 'lasot/LaSOTBenchmark',  # Prefix of image path
        pipeline=test_pipeline,  # Pipeline is passed by test_pipeline created before
        split='test',   # Split of dataset
        test_mode=True,  # Whether test mode
        only_eval_visible=True))  # whether to only evaluate the method on frames where the object is visible in LaSOT
# optimizer
optimizer = dict(  # Config used to build optimizer, support all the optimizers in PyTorch whose arguments are also the same as those in PyTorch
    type='SGD',
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys=dict(backbone=dict(lr_mult=0.1, decay_mult=1.0))))
optimizer_config = dict(  # Config used to build the optimizer hook, refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/optimizer.py#L8 for implementation details.
    type='SiameseRPNOptimizerHook',
    backbone_start_train_epoch=10,
    backbone_train_layers=['layer2', 'layer3', 'layer4'],
    grad_clip=dict(max_norm=10.0, norm_type=2))
# learning policy
lr_config = dict((  # Learning rate scheduler config used to register LrUpdater hook
    policy='SiameseRPN',
    lr_configs=[
        dict(type='step', start_lr_factor=0.2, end_lr_factor=1.0, end_epoch=5),
        dict(type='log', start_lr_factor=1.0, end_lr_factor=0.1, end_epoch=20),
    ])
# checkpoint saving
checkpoint_config = dict(interval=1)  # Config to set the checkpoint hook, Refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py for implementation.
evaluation = dict(
    metric=['track'],
    interval=1,
    start=10,
    rule='greater',
    save_best='success') # The config to build the evaluation hook
# yapf:disable
log_config = dict(  # config to register logger hook
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 20  # Total epochs to train the model
dist_params = dict(backend='nccl')  # Parameters to setup distributed training, the port is set to 29500 by default
log_level = 'INFO'  # The level of logging.
work_dir = './work_dirs/xxx'  # Directory to save the model checkpoints and logs for the current experiments.
load_from = None  # load models as a pre-trained model from a given path. This will not resume training.
resume_from = None  # Resume checkpoints from a given path, the training will be resumed from the epoch when the checkpoint's is saved.
workflow = [('train', 1)]  # Workflow for runner. [('train', 1)] means there is only one workflow and the workflow named 'train' is executed once. The workflow trains the model by 7 epochs according to the total_epochs.
```
