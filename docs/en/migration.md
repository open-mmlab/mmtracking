Compared with the 0.xx series of MMTracking, the latest 1.xx version of MMTracking has some importrant modifications.

## Configs

### file names
  
**old**: `deepsort_faster-rcnn_fpn_4e_mot17-private-half.py`

**new**: `deepsort_faster-rcnn-resnet50-fpn_8x2bs-4e_mot17halftrain_test-mot17halfval.py`

### keys of datasetloader

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

### The interface of train

### The interface of test


## Data

### data structure

The 1.xx versions of MMtracking add two new data structure: [TrackDataSample](mmtrack/structures/track_data_sample.py) and [ReIDDataSample](mmtrack/structures/reid_data_sample.py). These data structures wrap the annotations and predictions from one image (sequence) and used as interfaces between different components.

### dataset class

The 1.xx versions of MMTracking have two base dataset class which inheient from the `BaseDataset` in MMEngine: `BaseSOTDataset` and `BaseVideoDataset`. The former is only used in SOT and the latter is used for all other tasks.

### data pipeline

Most of the transforms on the image sequeces in the 0.xx versions are refactored. In the 1.xx versions of MMTracking, for the image sequences, we use `TransformBroadcaster` to wrap the transformes of single image.

Some tranformers on the images sequeces are reverved, such as `SeqCropLikeStark`, since `TransformBroadcaster` doesn't support setting different arguments respectively for each image in the sequece.


### data sampler

The 1.xx versions of MMtracking add `DATA_SAMPLERS` registry. You can customize different dataset samplers in the configs. Details about the samplers can be seen [here](mmtrack/datasets/samplers)

## Evaluation

The old version of MMTarcking implement evaluation in the dataset class. In the 1.xx versions of MMTracking, we add `METRICS` registry. All evaluation are implemented in the metric classes registered in `METRICS`. Details can be seen [here](mmtrack/evaluation/metrics).


## Visualization

The 1.xx versions of MMTracking add `TrackLocalVisualizer` and `DetLocalVisualizer` which are registered in `VISUALIZER`. Compared with the 0.xx versions of MMTracking, we support the visualization of images and feature maps. Details can be seen [here](mmtrack/visualization/local_visualizer.py)



## Engine

The runner of training, evaluation and test are upgraded in the 1.xx versions of MMTracking. Details can be seen in MMEngine.