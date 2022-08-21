#### 一个SiameseRPN++算法的配置文件例子

```python
cudnn_benchmark = True  ## 当设置为 True 时可以加快网络训练
crop_size = 511  # 边界框的裁剪大小
exemplar_size = 127  # 模板大小
search_size = 255  #  搜索大小

# 模型设置
model = dict(
    type='SiamRPN',  # 单目标跟踪器名称
    backbone=dict(  # 主干网络的配置
        type='SOTResNet',  # 主干网络的类型
        depth=50,  # 主干网络的深度，对于 ResNet 网络，通常使用50深度
        out_indices=(1, 2, 3),  # 每个阶段产生的输出特征图的索引
        frozen_stages=4,  # 被冻结的权重
        strides=(1, 2, 1, 1),  # 每个阶段卷积步长
        dilations=(1, 1, 2, 4),  # 每个阶段的空洞卷积
        norm_eval=True, # 是否冻结 BN 中的统计信息
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://download.openmmlab.com/mmtracking/pretrained_weights/sot_resnet50.model'  # noqa: E501
        )), # 主干网络的预训练权重
    neck=dict(
        type='ChannelMapper',  # 模型颈部名称.
        in_channels=[512, 1024, 2048],  # 输入通道
        out_channels=256,  # 输出通道
        kernel_size=1,  # 卷积核大小
        norm_cfg=dict(type='BN'),  # 标准化层配置
        act_cfg=None),  # 激活函数层配置
    head=dict(
        type='SiameseRPNHead',  # 模型头部名称
        anchor_generator=dict(  # anchor 生成器配置
            type='SiameseRPNAnchorGenerator',  # anchor生成器名称
            strides=[8],  # anchor 生成器的步长，这与 FPN 特征步长保持一致
            ratios=[0.33, 0.5, 1, 2, 3],  # 高宽比
            scales=[8]),  # anchor 的大小，特征图某一位置的锚点面积为 scale * base_sizes
        in_channels=[256, 256, 256],  # 输入通道，这与模型颈部的输出通道保持一致
        weighted_sum=True,  # 如果为 True，则使用可学习的权重对 siamese rpn 中的多个模型头部的输出进行加权求和，如果为 False，则取平均。
        bbox_coder=dict(  # 框编码器的配置，用于在训练和测试期间对框进行编码和解码
            type='DeltaXYWHBBoxCoder',  # 框编码器的名称. 'DeltaXYWHBBoxCoder' 被用于大部分方法。 详情请参考https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/coder/delta_xywh_bbox_coder.py#L9
            target_means=[0., 0., 0., 0.],  # 用于编码和解码框的均值
            target_stds=[1., 1., 1., 1.]),  # 用于编码和解码框的方差
        loss_cls=dict(  # 分类分支的损失函数配置
            type='CrossEntropyLoss', # 分类分支的损失函数类型，我们同样支持 Focal loss。
            reduction='sum',
            loss_weight=1.0),  # 分类分支的损失函数权重
        loss_bbox=dict(  # 回归分支的损失函数配置
            type='L1Loss',  # 损失函数名称，具体实现请查看 https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/losses/smooth_l1_loss.py#L56
            reduction='sum',
            loss_weight=1.2)),  # 回归分支损失函数权重
    train_cfg=dict(  # rpn 和 rcnn 训练时的超参数
        rpn=dict(  # rpn 训练时的配置
            assigner=dict(  # 分配器的配置
                type='MaxIoUAssigner',  # 分配器的名称， MaxIoUAssigner 被用在大量的检测器中，详情请参考 https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/assigners/max_iou_assigner.py#L10
                pos_iou_thr=0.6,  # IoU >= threshold 0.6 为正样本
                neg_iou_thr=0.3,  # IoU < threshold 0.3 为负样本
                min_pos_iou=0.6,  # 被选为正样本的最小的 IoU 阈值
                match_low_quality=False),  # 是否匹配低质量的框（有关更多详细信息，请参阅 API 文档）
            sampler=dict(  # 正负样本采样器配置
                type='RandomSampler',  # 采样器名称, PseudoSampler 和其它的采样器同样也支持，详情请参考 https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/samplers/random_sampler.py#L8
                num=64,  # 模板图像和搜索图像为正样本对的数目
                pos_fraction=0.25,  # 当模板图像和搜索图像为正样本对时，正样本在总样本中所占比例
                add_gt_as_proposals=False),  # 采样后是否添加真实标签框作为提案框
            num_neg=16,  # 示例图像和搜索图像为负样本对时的负样本数
            exemplar_size=exemplar_size,
            search_size=search_size)),
    test_cfg=dict(
        exemplar_size=exemplar_size,
        search_size=search_size,
        context_amount=0.5,  # 上下文数值
        center_size=7,  # 示例图像中心特征图裁剪后的大小
        rpn=dict(penalty_k=0.05, window_influence=0.42, lr=0.38)))  # 用于平滑预测后的图像

data_root = 'data/'  # 数据集路径
train_pipeline = [
    dict(
        type='PairSampling',  # 训练时的样本采样方法
        frame_range=5,  # 在和模板帧同一视频中搜索帧的采样范围
        pos_prob=0.8,  # 采样正样本对的概率
        filter_template_img=False), # 采样搜索帧时是否滤除模板帧
    dict(type='LoadMultiImagesFromFile',  # 第一步：从文件路径中载入多张图像
        to_float32=True),  # 将图片转换成 np.float32 格式
    dict(type='SeqLoadAnnotations',  # 第二步：载入图片的标注文件路径
        with_bbox=True),  # 是否使用边界框
    dict(
        type='SeqCropLikeSiamFC',  # 像 SiamFC 那样子裁剪图片
        context_amount=0.5,  # 上下文数值
        exemplar_size=exemplar_size,
        crop_size=crop_size),
    dict(
        type='SeqShiftScaleAug',  #  平移缩放图片
        target_size=[exemplar_size, search_size],  # 每张图片的目标大小
        shift=[4, 64],  # 每张图像的最大偏移量
        scale=[0.05, 0.18]),  # 每张图片最大缩放偏移量
    dict(type='SeqColorAug',  # 颜色增强
        prob=[1.0, 1.0]),  # 每张图像颜色增强的概率
    dict(type='SeqBlurAug',  # 模糊增强
        prob=[0.0, 0.2]),  # 每张图片模糊增强的概率
    dict(type='VideoCollect', keys=['img', 'gt_bboxes', 'is_positive_pairs']),  # 决定数据中哪些键应该传递给检测器
    dict(type='ConcatVideoReferences'),  # 拼接引用图像
    dict(type='SeqDefaultFormatBundle',  # 使用默认的方式格式化流水线中收集的数据
        ref_prefix='search')  # 参考图片中键值的前缀
]
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),  # 第一步：从文件路径中载入图像
    dict(type='LoadAnnotations', with_bbox=True), # 第二步：载入图片的标注文件路径
    dict(
        type='MultiScaleFlipAug',  # 测试阶段数据增强的封装
        scale_factor=1,  # 调整输入图像大小的比例因子
        flip=False,  # 测试时是否反转图像
        transforms=[
            dict(type='VideoCollect', keys=['img', 'gt_bboxes']),  # 测试时所需要的键
            dict(type='ImageToTensor', keys=['img'])  # 将图片转换成张量
        ])
]
# 数据集设置
data = dict(
    samples_per_gpu=28,  # 每个 GPU 中批量的大小
    workers_per_gpu=2,  # 为每个 GPU 预取数据的 Worker 的数目
    persistent_workers=True,  # persistent workers 设置
    samples_per_epoch=600000,  # 每个epoch的训练样本数量
    train=dict(
        type='RandomSampleConcatDataset',  # 随机采样联合数据集类
        dataset_sampling_weights=[0.25, 0.2, 0.55],  # 联合数据采样概率
        dataset_cfgs=[
            dict(
                type='SOTImageNetVIDDataset',  # 数据集类型
                ann_file=data_root +
                'ILSVRC/annotations/imagenet_vid_train.json',  # 标注文件路径
                img_prefix=data_root + 'ILSVRC/Data/VID',  # 图片路径前缀
                pipeline=train_pipeline,  # 训练的数据流水线
                split='train',  # 数据集划分集
                test_mode=False),  # 是否为测试模式
            dict(
                type='SOTCocoDataset',  # 数据集类型
                ann_file=data_root +
                'coco/annotations/instances_train2017.json',  # 标注文件路径
                img_prefix=data_root + 'coco/train2017',  # 图片路径前缀
                pipeline=train_pipeline,  # 训练的数据流水线
                split='train',  # 数据集划分集
                test_mode=False),  # 是否为测试模式
            dict(
                type='SOTCocoDataset',  # 数据集类型
                ann_file=data_root +
                'ILSVRC/annotations/imagenet_det_30plus1cls.json',  # 标注文件路径
                img_prefix=data_root + 'ILSVRC/Data/DET',  # 图片路径前缀
                pipeline=train_pipeline,  # 训练的数据流水线
                split='train',  # 数据集划分集
                test_mode=False)  # 是否为测试模式
        ]),
    val=dict(  # 验证集配置
        type='LaSOTDataset',  # 数据集类型
        ann_file=data_root + 'lasot/annotations/lasot_test_infos.txt',  # 数据集信息文件的路径
        img_prefix=data_root + 'lasot/LaSOTBenchmark',  # 图片路径前缀
        pipeline=test_pipeline,  # 验证时的流水线
        split='test',   # 数据集的划分子集
        test_mode=True,  # 是否为测试模式
        only_eval_visible=True),  # 在LaSOT上，是否仅在物体可见帧上评估方法
    test=dict(  # 测试集配置
        type='LaSOTDataset',  # 数据集类型
        ann_file=data_root + 'lasot/annotations/lasot_test_infos.txt',  # 数据集信息文件的路径
        img_prefix=data_root + 'lasot/LaSOTBenchmark'  # 图片路径前缀
        pipeline=test_pipeline,  # 测试时的流水线
        split='test',   # 数据集的划分子集
        test_mode=True,  # 是否为测试模式
        only_eval_visible=True))  # 在LaSOT上，是否仅在物体可见帧上评估方法
# 优化器
optimizer = dict(  # 优化器配置
    type='SGD',
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys=dict(backbone=dict(lr_mult=0.1, decay_mult=1.0))))
optimizer_config = dict(  # 优化器钩子配置， 详情请查看 https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/optimizer.py#L8
    type='SiameseRPNOptimizerHook',
    backbone_start_train_epoch=10,
    backbone_train_layers=['layer2', 'layer3', 'layer4'],
    grad_clip=dict(max_norm=10.0, norm_type=2))
# 训练策略
lr_config = dict((
    policy='SiameseRPN',
    lr_configs=[
        dict(type='step', start_lr_factor=0.2, end_lr_factor=1.0, end_epoch=5),
        dict(type='log', start_lr_factor=1.0, end_lr_factor=0.1, end_epoch=20),
    ])
# 模型权重文件
checkpoint_config = dict(interval=1)  # 模型权重文件配置，详情请查看 https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py
evaluation = dict(
    metric=['track'],
    interval=1,
    start=10,
    rule='greater',
    save_best='success') # 评测钩子配置
# yapf:disable
log_config = dict(  # 日志钩子配置
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# 运行时配置
total_epochs = 20  # 训练模型时总共的 epoch 数目
dist_params = dict(backend='nccl')  # 分布式训练后端，默认端口号为29500
log_level = 'INFO'  # 日志记录级别
work_dir = './work_dirs/xxx'  # 模型权重文件和日志保存的路径
load_from = None  # 从给定路径加载模型作为预训练模型
resume_from = None  # 从给定路径恢复模型权重文件
workflow = [('train', 1)]  # 训练器流程。 [('train', 1)] 表示只有一个名为 'train' 流程， 该流程将被执行一次
```
