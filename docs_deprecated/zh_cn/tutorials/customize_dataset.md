## 自定义数据集

对于自定义的数据集，你既可以将其转换成 CocoVID 格式，也可以实现一个全新的数据集。
对于 MMTracking，我们建议将数据离线转换成 CocoVID 格式，这样就可以直接使用 `CocoVideoDataset` 类了。 在这种情况下，你只需要修改配置文件中的数据标注路径和 `classes` 类别即可。

### 将数据集转换为 CocoVID 样式

#### CocoVID 标注文件

CocoVID 风格的标注文件需要以下键：

- `videos`：视频序列。每个视频都是包含 `name`、`id` 键的一个字典，键为 `name`、`id`。可选键有 `fps`、`width` 和 `height`。
- `images`：图像序列。每张图像都是包含 `file_name`、`height`、`width`、`id`、`frame_id` 和 `video_id` 键的一个字典。请注意，`frame_id` 的**索引是从 0 开始的**。
- `annotations`：实例标注序列。每个标注都是包含 `bbox`、`area`、`id`、`category_id`、`instance_id`、`image_id` 和 `video_id` 键的一个字典。其中 `instance_id` 仅用于跟踪任务。
- `categories`：类别序列。每个类别都是包含 `id`、`name` 键的一个字典。

[此处](https://github.com/open-mmlab/mmtracking/blob/master/tests/data/demo_cocovid_data/ann.json) 提供了一个简单实例。

[此处](https://github.com/open-mmlab/mmtracking/tree/master/tools/convert_datasets/) 提供了转换现有数据集的实例。

#### 修改配置

数据预处理后，用户需要进一步修改配置文件才能使用自定义数据集。
这里我们展示了一个使用 5 个类的自定义数据集的实例，假设其以及被转换成 CocoVID 格式。

在 `configs/my_custom_config.py`:

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

### 使用数据集包装器

MMTracking 还支持一些数据集包装器来混合数据集或修改数据集分布。
目前支持三个数据集包装器，如下所示：

- `RepeatDataset`：简单地重复整个数据集。
- `ClassBalancedDataset`：以类平衡的方式重复数据集。
- `ConcatDataset`：拼接多个数据集。

#### 重复数据集

我们使用 `RepeatDataset` 作为包装器来重复数据集。 例如，假设原始数据集是 `Dataset_A` ，我们需要重复该数据集，可以将配置做如下修改：

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

#### 类平衡数据集

我们使用 `ClassBalancedDataset` 作为包装器来实现类平衡数据集。
要重复的数据集需要实例化函数 `self.get_cat_ids(idx)` 来支持 `ClassBalancedDataset` 类。
例如，要使用配置 `oversample_thr=1e-3` 重复数据集 `Dataset_A`，配置如下所示：

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

#### 拼接数据集

有三种方法可以拼接数据集。

1. 如果要拼接的数据集类型相同，注释文件不同，可以将配置文件按照如下方式修改：

   ```python
   dataset_A_train = dict(
       type='Dataset_A',
       ann_file = ['anno_file_1', 'anno_file_2'],
       pipeline=train_pipeline
   )
   ```

   有两种方式支持测试或评估拼接后的数据集，默认的方式为对每个数据集进行单独评估。 如果你想要将拼接后的数据集作为一个整体进行评测，你可以设置 `separate_eval=False` 具体修改方式如下：

   ```python
   dataset_A_train = dict(
       type='Dataset_A',
       ann_file = ['anno_file_1', 'anno_file_2'],
       separate_eval=False,
       pipeline=train_pipeline
   )
   ```

2. 如果要连接的数据集类型不同，可以将配置文件按照如下方式修改：

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

   通过该方法拼接的数据集同样支持两种方式评测，默认的方式为对每个数据集进行单独评估。

3. 我们也支持显式定义 `ConcatDataset` 类。

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

   这种方式允许用户通过设置 `separate_eval=False` 来将所有数据集进行统一的评测。

**请注意：**

1. `separate_eval=False` 假设了数据集在评测使用 `self.data_infos` 方法。由于 `CocoVID` 数据集不完全依赖于 `self.data_infos` 方法进行评测。因此，`CocoVID` 数据集不支持将数据集统一起来评测。 同时，我们也不建议将不同类型的数据集并将它们作为一个整体进行评测。
2. 由于 `ClassBalancedDataset` 和 `RepeatDataset` 不支持评测，因此由这些数据集拼接而成的数据集也不支持评测。

这里有一个更复杂的例子：分别将 `Dataset_A` 和 `Dataset_B` 重复 N 次和 M 次，然后拼接重复的数据集，相关配置如下所示：

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

### 现有数据集的子集

对于现有的数据集，我们可以修改其配置文件中目标种类来训练子数据集。
例如，如果你只想训练当前数据集的三个类，
你可以将当前数据集配置文件中的目标中类做相应的修改。
数据集会自动过滤掉其他类的真实标签框。

```python
classes = ('person', 'bicycle', 'car')
data = dict(
    train=dict(classes=classes),
    val=dict(classes=classes),
    test=dict(classes=classes))
```

MMTracking 还支持从文件中读取类，这在实际应用中很常见。
例如，假设`classes.txt` 包含如下类名：

```
person
bicycle
car
```

用户可以将类设置为包含类的文件路径，数据集将加载并自动将其转换为列表。

```python
classes = 'path/to/classes.txt'
data = dict(
    train=dict(classes=classes),
    val=dict(classes=classes),
    test=dict(classes=classes))
```
