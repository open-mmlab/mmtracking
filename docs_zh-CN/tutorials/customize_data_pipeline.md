## 自定义数据预处理流程

MMTracking 中有两种数据流水线：

- 单张图片，这与 MMDetection 中的大部分情况相同
- 成对/多张图片

### 单张图片的数据预处理流程

对于单张图片，可以参考[MMDetection教程](https://mmdetection.readthedocs.io/en/latest/tutorials/data_pipeline.html)。
此外，MMTracking 还有些许不同之处：

- 我们的 `VideoCollect` 实现方法和 MMdetection 中的 `Collect` 相似，但是更适用于视频感知任务。例如：`frame_id` 和 `is_video_data` 属性会被默认收集出来。

### 多张图片的数据预处理流程

在多数情况下，我们需要同时处理多张图片。这主要是因为我们需要在同一视频中针对关键帧采样多个参考帧，以便于后续训练和推理。
请先察看单张图片的预处理实现，因为多张图片的预处理实现也是基于此。我们接下来将详细介绍整体流程。

#### 1. 采样参考帧

一旦我们得到关键帧的标注，我们将采样和加载参考帧的标注。

以 `CocoVideoDataset` 为例，我们用函数 `ref_img_sampling` 来采样和加载参考图片的标注。

```python
from mmdet.datasets import CocoDataset

class CocoVideoDataset(CocoDataset):

    def __init__(self,
                 ref_img_sampler=None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_img_sampler = ref_img_sampler


    def ref_img_sampling(self, **kwargs):
        pass

    def prepare_data(self, idx):
        img_info = self.data_infos[idx]
        if self.ref_img_sampler is not None:
            img_infos = self.ref_img_sampling(img_info, **self.ref_img_sampler)
        ...
```

在这种情况下，加载的标注不再是一个 `dict`, 而是一个包含关键图片和参考图片的 `list[dict]`。这个列表的首个元素就是关键图片的标注。

#### 2. 序列处理和收集数据

在这一步，我们执行图片转换并且收集图片信息。

单张图片预处理流程是接收字典作为输入，并输出字典，然后进入后续的图片转换; 与之不同，序列的预处理流程是接收一个包含字典的列表作为输入，并输出一个包含字典的列表，然后进入后续的图片转换。

序列的预处理流程通常继承于 MMDetection, 但是会对列表元素做循环处理。

```python
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadImageFromFile

@PIPELINES.register_module()
class LoadMultiImagesFromFile(LoadImageFromFile):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, results):
        outs = []
        for _results in results:
            _results = super().__call__(_results)
            outs.append(_results)
        return outs

```

有时，你需要增加参数 `share_params` 来决定是否共享图片转换的随机种子。

#### 3. 拼接参考图片（如果需要）

如果参考图片超过一个，我们利用 `ConcatVideoReferences` 来以字典形式收集所有参考图片。经处理后，该包含关键图片和参考图片的列表总长度为2。

#### 4. 将输出结果格式化成一个字典

最后，我们利用 `SeqDefaultFormatBundle` 来将数据的列表形式转换为字典形式，作为后续模型的输入。这里有一个数据全部处理流水线的示例：

```python
train_pipeline = [
    dict(type='LoadMultiImagesFromFile'),
    dict(type='SeqLoadAnnotations', with_bbox=True, with_track=True),
    dict(type='SeqResize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
    dict(type='SeqNormalize', **img_norm_cfg),
    dict(type='SeqPad', size_divisor=16),
    dict(
        type='VideoCollect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_instance_ids']),
    dict(type='ConcatVideoReferences'),
    dict(type='SeqDefaultFormatBundle', ref_prefix='ref')
]
```
