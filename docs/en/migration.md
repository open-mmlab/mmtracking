# Migration from MMTracking 0.xx

Compared with the 0.xx versions of MMTracking, the latest 1.xx version of MMTracking has the following important modifications.

## Overall Structures

The `core` in the old versions of MMTracking is splited into `engine`, `evaluation`, `structures`, `visualization` and `model/task_moduls` in the 1.xx version of MMTracking. Details can be seen in the [user guides](../../docs/en/user_guides).

## Configs

### file names

**old**: `deepsort_faster-rcnn_fpn_4e_mot17-private-half.py`

**new**: `deepsort_faster-rcnn-resnet50-fpn_8x2bs-4e_mot17halftrain_test-mot17halfval.py`

### keys of dataset loader

**old**

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

**new**

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

### keys of optimizer

**old**

```python
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys=dict(backbone=dict(lr_mult=0.1, decay_mult=1.0))))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
```

**new**

```python
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys=dict(backbone=dict(lr_mult=0.1, decay_mult=1.0))))
```

### keys of learning scheduler

**old**

```python
lr_config = dict(policy='step', step=[400])
```

**new**

```python
param_scheduler = dict(type='MultiStepLR', milestones=[400], gamma=0.1)
```

## Model

### Data preprocessor

The 1.xx versions of MMtracking add [TrackDataPreprocessor](../../mmtrack/models/data_preprocessors/data_preprocessor.py). The data out from the data pipeline is transformed by this module and then fed into the model.

### Train

The training forward of models and heads is performed by calling `loss` function in their respective classes. The arguments of `loss` function in models contain a dict of `Tensor` and a list of `TrackDataSample`.

### Test

The test forward of models and heads is performed by calling `predict` function in their respective classes. The arguments of `predict` function in models contain a dict of `Tensor` and a list of `TrackDataSample`.

## Data

### data structure

The 1.xx versions of MMtracking add two new data structure: [TrackDataSample](../../mmtrack/structures/track_data_sample.py) and [ReIDDataSample](../../mmtrack/structures/reid_data_sample.py). These data structures wrap the annotations and predictions from one image (sequence) and are used as interfaces between different components.

### dataset class

The 1.xx versions of MMTracking add two base dataset classes which inherient from the `BaseDataset` in MMEngine: `BaseSOTDataset` and `BaseVideoDataset`. The former is only used in SOT and the latter is used for all other tasks.

### data pipeline

1. Most of the transforms on image sequences in the old MMTracking are refactored in the latest MMTracking. Specifically, we use `TransformBroadcaster` to wrap the transforms of single image.

Some transforms on image sequences, such as `SeqCropLikeStark`, are reserved since `TransformBroadcaster` doesn't support setting different arguments respectively for each image in the sequence.

2. We pack the `VideoCollect`, `ConcatSameTypeFrames` and `SeqDefaultFormatBundle` in the old MMTracking into `PackTrackInputs` in the latest MMTracking.

3. The normalizaion in the pipeline in the old MMTracking is removed and this operation is implemented in the model forward.

### data sampler

The 1.xx versions of MMtracking add `DATA_SAMPLERS` registry. You can customize different dataset samplers in the configs. Details about the samplers can be seen [here](../../mmtrack/datasets/samplers).

## Evaluation

The old versions of MMTarcking implement evaluation in the dataset class. In the 1.xx versions of MMTracking, we add `METRICS` registry. All evaluation are implemented in the metric classes registered in `METRICS`. Details can be seen [here](../../mmtrack/evaluation/metrics).

## Visualization

The 1.xx versions of MMTracking add `TrackLocalVisualizer` and `DetLocalVisualizer` which are registered in `VISUALIZER`. Compared with the 0.xx versions of MMTracking, we support the visualization of images and feature maps. Details can be seen [here](../../mmtrack/visualization/local_visualizer.py).

## Engine

The runner, hook, logging and optimizer in the training, evaluation and test are refactored in the 1.xx versions of MMTracking. Details can be seen in MMEngine.
