## 了解配置文件

我们使用 python 文件作为我们的配置系统。 你可以在 $MMTracking/configs 下找到所有配置文件。

我们将模块化和继承融入我们的配置系统，这将方便我们进行各种实验。
你可以运行 `python tools/print_config.py /PATH/TO/CONFIG` 来查看完整的配置。

### 通过脚本参数修改配置

使用 “tools/train.py” 或 “tools/test.py” 提交任务时，你可以指定 `--cfg-options` 来原地修改配置。

- 更新配置文件中字典的键值。

  可以按照原始配置中字典键的顺序修改配置选项。
  例如 `--cfg-options model.detector.backbone.norm_eval=False` 将模型骨干网络中的 BN 模块更改为训练模式。

- 更新配置文件列表中的字典键值。

  在配置文件中，一些配置字典被组成一个列表。例如，测试时的流水线 `data.test.pipeline` 通常是一个列表
  例如 `[dict(type='LoadImageFromFile'), ...]` 。如果你想把流水线中的 `'LoadImageFromFile'` 更改为 `'LoadImageFromWebcam'`，
  你可以使用 `--cfg-options data.test.pipeline.0.type=LoadImageFromWebcam`。

- 更新列表/元组的值。

  如果要更新的值是列表或元组。例如配置文件中通常设置 `workflow=[('train', 1)]` 。如果你想
  改变这个键值，你可以指定 `--cfg-options workflow="[(train,1),(val,1)]"` 。注意引号"是为了
  支持列表/元组数据类型，并且在指定键值的引号内 **不允许** 有空格。

### 配置文件结构

`config/_base_` 下有3种基本组件类型分别是 dataset，model，default_runtime。
许多方法可以通过这些文件构筑，例如 DFF、FGFA、SELSA、SORT、DeepSORT。
由来自 `_base_` 的组件组成的配置称为 _primitive_ 。

对于同一文件夹下的所有配置，建议只有**一个** _primitive_ 配置。所有其他配置都应该从 _primitive_ 配置中继承。在这种方式下，最大的继承级别是3。

为了便于理解，我们建议开发者继承现有方法。
例如，如果基于 Faster R-CNN 做了一些修改，用户可以先通过指定 `_base_ = ../../_base_/models/faster_rcnn_r50_dc5.py` 来继承基本的 Faster R-CNN 结构，然后在配置文件中修改必要的字段。

如果您正在构建一个与任何现有方法都不共享结构的新方法，你可以在 `configs` 下创建一个文件夹 `xxx_rcnn`，

详细文档请参考[mmcv](https://mmcv.readthedocs.io/en/latest/understand_mmcv/config.html#config)。

### 配置文件名样式

我们遵循以下样式来命名配置文件。建议开发者遵循相同的样式。

```
{model}_[model setting]_{backbone}_{neck}_[norm setting]_[misc]_[gpu x batch_per_gpu]_{schedule}_{dataset}
```

`{xxx}` 是必填字段，`[yyy]` 是可选字段。

- `{model}`：模型类型，例如 `dff`、`tracktor`、`siamese_rpn` 等。
- `[model setting]`：某些模型中的特定设置，例如 `dff`、`tracktor` 中使用的 `faster_rcnn`。
- `{backbone}`：主干网络，例如 `r50` (ResNet-50)、`x101` (ResNeXt-101)。
- `{neck}`：模型颈部，例如 `fpn`、`c5`。
- `[norm_setting]`：标准化设置，除了`bn`（Batch Normalization）不需要注明以外，其他标准化类型比如`gn`（Group Normalization），`syncbn`（Synchronized Batch Normalization）都应注明。
  `gn-head`/`gn-neck` 表示 GN 仅应用于模型头部/模型颈部，而 `gn-all` 表示 GN 应用于整个模型，例如主干网络，模型颈部，模型头部。
- `[misc]`：模型的其他设置/插件，例如`dconv`、`gcb`、`attention`、`albu`、`mstrain`。
- `[gpu x batch_per_gpu]`：GPU 数目以及每个 GPU 的样本数，默认使用 `8x2`。
- `{schedule}`：训练时长，选项为 `4e`、`7e`、`20e` 等。
  `20e` 表示 20 个周期。
- `{dataset}`：数据集如 `imagenetvid`、`mot17`、`lasot`。

### 配置文件详细解析

不同任务的配置文件结构请参考相应页面。

[视频目标检测](https://mmtracking.readthedocs.io/zh_CN/latest/tutorials/config_vid.html)

[多目标跟踪](https://mmtracking.readthedocs.io/zh_CN/latest/tutorials/config_mot.html)

[单目标跟踪](https://mmtracking.readthedocs.io/zh_CN/latest/tutorials/config_sot.html)

### 常见问题

#### 忽略基础配置中的一些字段

你可以设置 `_delete_=True` 来忽略基本配置中的某些字段。
详细内容请参考 [mmcv](https://mmcv.readthedocs.io/en/latest/understand_mmcv/config.html#inherit-from-base-config-with-ignored-fields)

#### 使用配置中的中间变量

配置文件中使用了一些中间变量，例如数据集中的 `train_pipeline`/`test_pipeline`。
值得注意的是，在子配置中修改中间变量时，用户需要再次将中间变量传递到相应的字段中。
例如，我们想使用自适应步长的测试策略来测试 SELSA。 ref_img_sampler 是我们想要修改的中间变量。

```python
_base_ = ['./selsa_faster_rcnn_r50_dc5_1x_imagenetvid.py']

# dataset settings
ref_img_sampler = dict(
    _delete_=True,
    num_ref_imgs=14,
    frame_range=[-7, 7],
    method='test_with_adaptive_stride')
data = dict(
    val=dict(
        ref_img_sampler=ref_img_sampler),
    test=dict(
        ref_img_sampler=ref_img_sampler))
```

我们首先需要定义新的 `ref_img_sampler` 然后将其赋值到 `data` 字段下对应位置。
