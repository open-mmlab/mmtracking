## Customize Data Pipelines

There are two types of data pipelines in MMTracking:

- Single image, which is consistent with MMDetection in most cases.
- Pair-wise / multiple images.

### Data pipeline for a single image

For a single image, you may refer to the [tutorial in MMDetection](https://mmdetection.readthedocs.io/en/latest/tutorials/data_pipeline.html).

There are several differences in MMTracking:

- We implement `VideoCollect` which is similar to `Collect` in MMDetection but is more compatible with the video perception tasks. For example, the meta keys `frame_id` and `is_video_data` are collected by default.

### Data pipeline for multiple images

In some cases, we may need to process multiple images simultaneously.
This is basically because we need to sample reference images of the key image in the same video to facilitate the training or inference process.

Please firstly take a look at the case of a single images above because the case of multiple images is heavily rely on it.
We explain the details of the pipeline below.

#### 1. Sample reference images

We sample and load the annotations of the reference images once we get the annotations of the key image.

Take `CocoVideoDataset` as an example, there is a function `sample_ref_img` to sample and load the annotations of the reference images.

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

In this case, the loaded annotations is no longer a `dict` but `list[dict]` that contains the annotations for the key and reference images.
The first item of the list indicates the annotations of the key image.

#### 2. Sequentially process and collect the data

In this step, we apply the transformations and then collected the information of the images.

In contrast to the pipeline of a single image that take a dictionary as the input and also output a dictionary for the next transformation, the sequential pipelines take a list of dictionaries as the input and also output a list of dictionaries for the next transformation.

These sequential pipelines are generally inherited from the pipeline in MMDetection but process the list in a loop.

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

Sometimes you may need to add a parameter `share_params` to decide whether share the random seed of the transformation on these images.

#### 3. Concat the reference images (if applicable)

If there are more than one reference image, we implement `ConcatVideoReferences` to collect the reference images to a dictionary.
The length of the list is 2 after the process.

#### 4. Format the output to a dictionary

In the end, we implement `SeqDefaultFormatBundle` to convert the list to a dictionary as the input of the model forward.

Here is an example of the data pipeline:

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
