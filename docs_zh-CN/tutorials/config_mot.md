#### 一个Tracktor算法的配置文件例子

#### An example of Tracktor

```python
model = dict(
    type='Tracktor',  # 多目标跟踪器的名称
    detector=dict(  # 检测器的详细配置说明请查看 https://github.com/open-mmlab/mmdetection/blob/master/docs_zh-CN/tutorials/config.md#mask-r-cnn-配置文件示例
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
        ), # 检测器预训练权重，它也会在测试中使用
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
    reid=dict(  # 重识别模型的配置
        type='BaseReID',  # 重识别模型的名称
        backbone=dict(  # 重识别模型的主干网络配置
            type='ResNet', # 详细请查看 https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/backbones/resnet.py#L288 了解更多的主干网络
            depth=50,  # 主干网络的深度，对于 ResNet 以及 ResNext 网络，通常使用50或者101深度
            num_stages=4,  # 主干网络中阶段的数目
            out_indices=(3, ),  # 每个阶段产生的输出特征图的索引
            style='pytorch'),  # 主干网络的形式，'pytorch' 表示步长为2的网络层在3x3的卷积中，'caffe' 表示步长为2的网络层在1x1卷积中。
        neck=dict(type='GlobalAveragePooling', kernel_size=(8, 4), stride=1),  # 重识别模型的颈部,通常是全局池化层。
        head=dict(  # 重识别模型的头部
            type='LinearReIDHead',  # 分类模型头部的名称
            num_fcs=1,  # 模型头部的全连接层数目
            in_channels=2048,  # 输入通道的数目
            fc_channels=1024,  # 全连接层通道数目
            out_channels=128,  # 输出通道数目
            norm_cfg=dict(type='BN1d'),  # 规一化模块的配置
            act_cfg=dict(type='ReLU')),  # 激活函数模块的配置
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://download.openmmlab.com/mmtracking/mot/reid/reid_r50_6e_mot17-4bf6b63d.pth'  # noqa: E501
        )), # 重识别模型预训练权重，它也会在测试中使用
    motion=dict(  # 运动模型配置
        type='CameraMotionCompensation',  # 运动模型名称
        warp_mode='cv2.MOTION_EUCLIDEAN',  # 包装模式
        num_iters=100, # 迭代次数
        stop_eps=1e-05),  # 停止迭代阈值
    tracker=dict(  # 跟踪器配置
        type='TracktorTracker',  # 跟踪器名称
        obj_score_thr=0.5,  # 检测目标的分类分数阈值
        regression=dict(  # Tracktor 跟踪器的回归模块
            obj_score_thr=0.5,  # 检测目标的分类分数阈值
            nms=dict(type='nms', iou_threshold=0.6),  # 回归器非极大值抑制配置
            match_iou_thr=0.3),  # 检测目标框的交并比阈值
        reid=dict(  # 测试阶段重识别模块配置
            num_samples=10,  # 计算特征相似性的最大样本数目
            img_scale=(256, 128),  # 重识别模型输入的图片大小
            img_norm_cfg=None,  # 重识别网络输入的标准化配置，None 表示与主干网络一致
            match_score_thr=2.0,  # 特征相似性阈值
            match_iou_thr=0.2),  # 交并比匹配阈值
        momentums=None,  # 更新缓冲区的动量
        num_frames_retain=10))  # 保留消失轨迹的最大帧数
# 以下配置与视频目标检测一致。 详情请参考 `config_vid.md`
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
