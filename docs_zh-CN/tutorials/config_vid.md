#### 一个DFF算法的配置文件例子

```python
model = dict(
    type='DFF',  # 视频目标检测器名称
    detector=dict(  # 详情请参考 https://mmdetection.readthedocs.io/zh_CN/latest/tutorials/config.html#mask-r-cnn
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
        type='FlowNetSimple',  # 运动模型名称
        img_scale_factor=0.5, # 对输入运动模型的图像进行下采样/上采样的比例因子
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://download.openmmlab.com/mmtracking/pretrained_weights/flownet_simple.pth'  # noqa: E501
        )), # 预训练模型权重
    train_cfg=None,
    test_cfg=dict(key_frame_interval=10))  # 测试时关键帧间隔
dataset_type = 'ImagenetVIDDataset'  # 数据集类型
data_root = 'data/ILSVRC/'  # 数据集根目录
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],  # 均值
    std=[58.395, 57.12, 57.375],  # 方差
    to_rgb=True)  # 图像通道顺序
train_pipeline = [  # 训练流水线
    dict(type='LoadMultiImagesFromFile'),  # 第一步：从文件路径中载入多张图像
    dict(
        type='SeqLoadAnnotations',  # 第二步：载入图片的标注文件路径
        with_bbox=True,  # 是否使用边界框
        with_track=True),  # 是否使用实例标签
    dict(type='SeqResize',   # 调整图片大小
        img_scale=(1000, 600),  # 最大的图像尺寸
        keep_ratio=True),  # 是否保持宽高比
    dict(
        type='SeqRandomFlip',  # 反转图片
        share_params=True,
        flip_ratio=0.5),  # 反转图片的概率
    dict(
        type='SeqNormalize',  # 标准化输入图片
        mean=[123.675, 116.28, 103.53],  # 与 img_norm_cfg 字段相同
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='SeqPad',  # 填充图片
        size_divisor=16),  # 填充后图像的边长需要被 size_divisor 整除
    dict(
        type='VideoCollect',  # 决定数据中哪些键应该传递给检测器
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_instance_ids']),
    dict(type='ConcatVideoReferences'),  # 拼接引用图像
    dict(type='SeqDefaultFormatBundle',  # 使用默认的方式格式化流水线中收集的数据
        ref_prefix='ref')  # 引用图片中键的前缀
]
test_pipeline = [
    dict(type='LoadImageFromFile'),  # 第一步：从文件路径中载入图像
    dict(
        type='MultiScaleFlipAug',  # 测试阶段数据增强的封装
        img_scale=(1000, 600),  # 测试图片的最大尺寸，通常用于调整大小的流水线方法
        flip=False,  # 测试时，是否反转图像
        transforms=[
            dict(type='Resize',  # 调整图像大小的数据增强方式
                keep_ratio=True),  # 是否保持高宽比例，这里设置的 img_scale 优先级低于上面的 img_scale
            dict(type='RandomFlip'),  # 因为 flip=False 所以 RandomFlip 没有用处
            dict(
                type='Normalize',  # 归一化设置, 其数值来自 img_norm_cfg
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=16), # 填充图片使其边长能被 size_divisor 整除
            dict(type='ImageToTensor', keys=['img']),  # 将图片转换成张量
            dict(type='VideoCollect', keys=['img'])  # 收集流水线中必要的键用于测试
        ])
]
data = dict(
    samples_per_gpu=1,  # 每个 GPU 中批量的大小
    workers_per_gpu=2,  # 为每个 GPU 预取数据的 Worker 的数目
    train=[
        dict(  # 训练集配置
            type='ImagenetVIDDataset',  # 数据集类型
            ann_file='data/ILSVRCannotations/imagenet_vid_train.json',  # 标注文件路径
            img_prefix='data/ILSVRCData/VID',  # 图片路径前缀
            ref_img_sampler=dict(  # # 采样引用图片的配置
                num_ref_imgs=1,
                frame_range=9,
                filter_key_img=False,
                method='uniform'),
            pipeline=train_pipeline),  # 训练流水线
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
    val=dict(  # 验证集配置
        type='ImagenetVIDDataset',
        ann_file='data/ILSVRCannotations/imagenet_vid_val.json',
        img_prefix='data/ILSVRCData/VID',
        ref_img_sampler=None,
        pipeline=test_pipeline,  # 验证时的流水线
        test_mode=True),
    test=dict(  # 测试集配置
        type='ImagenetVIDDataset',
        ann_file='data/ILSVRCannotations/imagenet_vid_val.json',
        img_prefix='data/ILSVRCData/VID',
        ref_img_sampler=None,
        pipeline=test_pipeline,  # 测试时的流水线
        test_mode=True))
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)  # 优化器配置
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))  # 优化器钩子配置， 详情请查看 https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/optimizer.py#L8
checkpoint_config = dict(interval=1)  # 模型权重文件配置，详情请查看 https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])  # 日志钩子配置
dist_params = dict(backend='nccl', port='29500') # 分布式训练后端，默认端口号为29500
log_level = 'INFO'  # 日志记录级别
load_from = None  # 从给定路径加载模型作为预训练模型
resume_from = None  # 从给定路径恢复模型权重文件
workflow = [('train', 1)]  # 训练器流程。 [('train', 1)] 表示只有一个名为 'train'流程， 该流程将被执行一次
lr_config = dict(  # 训练策略
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
    step=[2, 5])
total_epochs = 7  # 训练模型时总共的 epoch 数目
evaluation = dict(metric=['bbox'], interval=7)  # 评测钩子配置
work_dir = '../mmtrack_output/tmp'  # 模型权重文件和日志保存的路径
```
