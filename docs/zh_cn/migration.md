# MMTracking 版本迁移

与 0.xx 版本的 MMTracking 相比, 最新的 1.xx 版本有以下重要修改。

## 整体架构

MMTracking 0.xx 版本中的 `core`，在 1.xx 的新版中被划分为了 `engine`、 `evaluation`、 `structures`、 `visualization` 和 `model/task_moduls` 模块。详细信息可见[用户指南](../../docs/en/user_guides)。

## 配置

### 文件名

**旧版:** `deepsort_faster-rcnn_fpn_4e_mot17-private-half.py`

**新版:** `deepsort_faster-rcnn-resnet50-fpn_8x2bs-4e_mot17halftrain_test-mot17halfval.py`

### Dataset Loader  配置变更

**旧版**

```python
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    persistent_workers=True,
    samples_per_epoch=60000,
    train=dict(
        type='GOT10kDataset',
        ann_file=data_root +
        'got10k/annotations/got10k_train_infos.txt',
        img_prefix=data_root + 'got10k',
        pipeline=train_pipeline,
        split='train',
        test_mode=False),
    val=dict(
        type='GOT10kDataset',
        ann_file=data_root + 'got10k/annotations/got10k_test_infos.txt',
        img_prefix=data_root + 'got10k',
        pipeline=test_pipeline,
        split='test',
        test_mode=True),
    test=dict(
        type='GOT10kDataset',
        ann_file=data_root + 'got10k/annotations/got10k_test_infos.txt',
        img_prefix=data_root + 'got10k',
        pipeline=test_pipeline,
        split='test',
        test_mode=True))
```

**新版**

```python
train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='QuotaSampler', samples_per_epoch=60000),
    dataset=dict(
        type='GOT10kDataset',
        data_root=data_root,
        ann_file='GOT10k/annotations/got10k_train_infos.txt',
        data_prefix=dict(img_path='GOT10k'),
        pipeline=train_pipeline,
        test_mode=False))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='VideoSampler'),
    dataset=dict(
        type='GOT10kDataset',
        data_root='data/',
        ann_file='GOT10k/annotations/got10k_test_infos.txt',
        data_prefix=dict(img_path='GOT10k'),
        pipeline=test_pipeline,
        test_mode=True))
test_dataloader = val_dataloader
```

### Optimizer  配置变更

**旧版**

```python
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys=dict(backbone=dict(lr_mult=0.1, decay_mult=1.0))))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
```

**新版**

```python
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys=dict(backbone=dict(lr_mult=0.1, decay_mult=1.0))))
```

### Learning Scheduler  配置变更

**旧版**

```python
lr_config = dict(policy='step', step=[400])
```

**新版**

```python
param_scheduler = dict(type='MultiStepLR', milestones=[400], gamma=0.1)
```

## 模型

### 数据预处理

MMTracking 的 1.xx 版本增加了 [TrackDataPreprocessor](../../mmtrack/models/data_preprocessors/data_preprocessor.py)。由数据 pipeline 输出的数据经该模块的转换后，再被送入到模型中。

### 训练

在训练阶段的正向传播过程中，Models 和 Heads 类会通过 `loss` 函数来计算 loss。 `loss` 函数的输入包括一个值为 `Tensor` 的字典和一个元素为 `TrackDataSample` 的列表。

### 测试

在测试阶段的正向传播过程中，Models 及 Heads 会通过调用 `predict` 函数来得到预测结果。 `predict` 函数的输入包括一个值为 `Tensor` 的字典和一个元素为 `TrackDataSample` 的列表。

## 数据

### 数据结构

MMTracking 的 1.xx 版本新增了两个数据结构: [TrackDataSample](../../mmtrack/structures/track_data_sample.py) 和 [ReIDDataSample](../../mmtrack/structures/reid_data_sample.py)。这些数据结构封装了一张图片或序列的真值标签及预测结果，并将其用作为不同组件的接口。

### Dataset 类

MMTracking 的 1.xx 版本新增了两个继承自 MMEngine 的 Dataset 基类： `BaseSOTDataset` 和 `BaseVideoDataset`。前者仅用于单目标跟踪（SOT），后者可用于其余所有任务。

### Data Pipeline

1. MMTracking 旧版中的大部分图像序列变换，在新版本中被重构。具体来说，我们使用 `TransformBroadcaster` 来封装单个图像的变换。

   我们仍旧保留了某些图像序列的变换，例如 `SeqCropLikeStark`，因为 `TransformBroadcaster` 不支持分别为序列中的每个图像设置不同的参数。

2. 我们将旧版的 `VideoCollect`, `ConcatSameTypeFrames` 和 `SeqDefaultFormatBundle` 进行了封装 ，在新版中使用 `PackTrackInputs`。

3. 旧版本 pipeline 中的 normalizaion 被移除，其操作移至模型的正向传播过程。

### 数据采样

MMTracking 的 1.xx 版本新增了 `DATA_SAMPLERS` 注册器。您可以在配置中自定义不同的数据集采样器，详细信息参见[此处](../../mmtrack/datasets/samplers)。

## 评估

在旧版本的 MMTarcking 中，评估过程使用的是 Dataset 类。在 MMTracking 的 1.xx 版本中，我们新增了 `METRICS` 注册器，所有的评估都在 metric 类中实现，它们都注册在 `METRICS` 中，详细信息参见[此处](../../mmtrack/evaluation/metrics)。

## 可视化

新增了 `TrackLocalVisualizer` 和 `DetLocalVisualizer`，它们被注册进 `VISUALIZER` 注册器中。相较于旧版，新版支持了图像和特征图的可视化，详细信息参见[此处](../../mmtrack/visualization/local_visualizer.py)。

## 引擎

MMTracking 的 1.xx 版本重构了 `runner`、 `hook`、 `logging` 和 `optimizer`，详细信息可见于 MMEngine。
