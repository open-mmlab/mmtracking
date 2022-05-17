## Customize Datasets

To customize a new dataset, you can convert them to the existing CocoVID style or implement a totally new dataset.
In MMTracking, we recommend to convert the data into CocoVID style and do the conversion offline, thus you can use the `CocoVideoDataset` directly. In this case, you only need to modify the config's data annotation paths and the `classes`.

### Convert the dataset into CocoVID style

#### The CocoVID annotation file

The annotation json files in CocoVID style has the following necessary keys:

- `videos`: contains a list of videos. Each video is a dictionary with keys `name`, `id`. Optional keys include `fps`, `width`, and `height`.
- `images`: contains a list of images. Each image is a dictionary with keys `file_name`, `height`, `width`, `id`, `frame_id`, and `video_id`. Note that the `frame_id` is **0-index** based.
- `annotations`: contains a list of instance annotations. Each annotation is a dictionary with keys `bbox`, `area`,  `id`, `category_id`, `instance_id`, `image_id` and `video_id`. The `instance_id` is only required for tracking.
- `categories`: contains a list of categories. Each category is a dictionary with keys `id` and `name`.

A simple example is presented at [here](https://github.com/open-mmlab/mmtracking/blob/master/tests/data/demo_cocovid_data/ann.json).

The examples of converting existing datasets are presented at [here](https://github.com/open-mmlab/mmtracking/tree/master/tools/convert_datasets/).

#### Modify the config

After the data pre-processing, the users need to further modify the config files to use the dataset.
Here we show an example of using a custom dataset of 5 classes, assuming it is also in CocoVID format.

In `configs/my_custom_config.py`:

```python
...
# dataset settings
dataset_type = 'CocoVideoDataset'
classes = ('a', 'b', 'c', 'd', 'e')
...
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file='path/to/your/train/data',
        ...),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file='path/to/your/val/data',
        ...),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file='path/to/your/test/data',
        ...))
...
```

### Using dataset wrappers

MMTracking also supports some dataset wrappers to mix the dataset or modify the dataset distribution for training.
Currently it supports to three dataset wrappers as below:

- `RepeatDataset`: simply repeat the whole dataset.
- `ClassBalancedDataset`: repeat dataset in a class balanced manner.
- `ConcatDataset`: concat datasets.

#### Repeat dataset

We use `RepeatDataset` as wrapper to repeat the dataset. For example, suppose the original dataset is `Dataset_A`, to repeat it, the config looks like the following

```python
dataset_A_train = dict(
        type='RepeatDataset',
        times=N,
        dataset=dict(  # This is the original config of Dataset_A
            type='Dataset_A',
            ...
            pipeline=train_pipeline
        )
    )
```

#### Class balanced dataset

We use `ClassBalancedDataset` as wrapper to repeat the dataset based on category
frequency. The dataset to repeat needs to instantiate function `self.get_cat_ids(idx)`
to support `ClassBalancedDataset`.
For example, to repeat `Dataset_A` with `oversample_thr=1e-3`, the config looks like the following

```python
dataset_A_train = dict(
        type='ClassBalancedDataset',
        oversample_thr=1e-3,
        dataset=dict(  # This is the original config of Dataset_A
            type='Dataset_A',
            ...
            pipeline=train_pipeline
        )
    )
```

#### Concatenate dataset

There are three ways to concatenate the dataset.

1. If the datasets you want to concatenate are in the same type with different annotation files, you can concatenate the dataset configs like the following.

   ```python
   dataset_A_train = dict(
       type='Dataset_A',
       ann_file = ['anno_file_1', 'anno_file_2'],
       pipeline=train_pipeline
   )
   ```

   If the concatenated dataset is used for test or evaluation, this manner supports to evaluate each dataset separately. To test the concatenated datasets as a whole, you can set `separate_eval=False` as below.

   ```python
   dataset_A_train = dict(
       type='Dataset_A',
       ann_file = ['anno_file_1', 'anno_file_2'],
       separate_eval=False,
       pipeline=train_pipeline
   )
   ```

2. In case the dataset you want to concatenate is different, you can concatenate the dataset configs like the following.

   ```python
   dataset_A_train = dict()
   dataset_B_train = dict()

   data = dict(
       imgs_per_gpu=2,
       workers_per_gpu=2,
       train = [
           dataset_A_train,
           dataset_B_train
       ],
       val = dataset_A_val,
       test = dataset_A_test
       )
   ```

   If the concatenated dataset is used for test or evaluation, this manner also supports to evaluate each dataset separately.

3. We also support to define `ConcatDataset` explicitly as the following.

   ```python
   dataset_A_val = dict()
   dataset_B_val = dict()

   data = dict(
       imgs_per_gpu=2,
       workers_per_gpu=2,
       train=dataset_A_train,
       val=dict(
           type='ConcatDataset',
           datasets=[dataset_A_val, dataset_B_val],
           separate_eval=False))
   ```

   This manner allows users to evaluate all the datasets as a single one by setting `separate_eval=False`.

**Note:**

1. The option `separate_eval=False` assumes the datasets use `self.data_infos` during evaluation. Therefore, CocoVID datasets do not support this behavior since CocoVID datasets do not fully rely on `self.data_infos` for evaluation. Combining different types of datasets and evaluating them as a whole is not tested thus is not suggested.
2. Evaluating `ClassBalancedDataset` and `RepeatDataset` is not supported thus evaluating concatenated datasets of these types is also not supported.

A more complex example that repeats `Dataset_A` and `Dataset_B` by N and M times, respectively, and then concatenates the repeated datasets is as the following.

```python
dataset_A_train = dict(
    type='RepeatDataset',
    times=N,
    dataset=dict(
        type='Dataset_A',
        ...
        pipeline=train_pipeline
    )
)
dataset_A_val = dict(
    ...
    pipeline=test_pipeline
)
dataset_A_test = dict(
    ...
    pipeline=test_pipeline
)
dataset_B_train = dict(
    type='RepeatDataset',
    times=M,
    dataset=dict(
        type='Dataset_B',
        ...
        pipeline=train_pipeline
    )
)
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train = [
        dataset_A_train,
        dataset_B_train
    ],
    val = dataset_A_val,
    test = dataset_A_test
)

```

### Subset of existing datasets

With existing dataset types, we can modify the class names of them to train subset of the annotations.
For example, if you want to train only three classes of the current dataset,
you can modify the classes of dataset.
The dataset will filter out the ground truth boxes of other classes automatically.

```python
classes = ('person', 'bicycle', 'car')
data = dict(
    train=dict(classes=classes),
    val=dict(classes=classes),
    test=dict(classes=classes))
```

MMTracking also supports to read the classes from a file, which is common in real applications.
For example, assume the `classes.txt` contains the name of classes as the following.

```
person
bicycle
car
```

Users can set the classes as a file path, the dataset will load it and convert it to a list automatically.

```python
classes = 'path/to/classes.txt'
data = dict(
    train=dict(classes=classes),
    val=dict(classes=classes),
    test=dict(classes=classes))
```
