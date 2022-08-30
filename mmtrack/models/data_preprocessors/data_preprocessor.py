# Copyright (c) OpenMMLab. All rights reserved.
from numbers import Number
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from mmdet.structures.mask import BitmapMasks
from mmengine.model import BaseDataPreprocessor

from mmtrack.registry import MODELS
from mmtrack.structures import TrackDataSample
from mmtrack.utils import stack_batch


@MODELS.register_module()
class TrackDataPreprocessor(BaseDataPreprocessor):
    """Image pre-processor for tracking tasks.

    Accepts the data sampled by the dataloader, and preprocesses it into the
    format of the model input. ``TrackDataPreprocessor`` provides the
    tracking data pre-processing as follows:

    - Collate and move data to the target device.
    - Pad inputs to the maximum size of current batch with defined
      ``pad_value``. The padding size can be divisible by a defined
      ``pad_size_divisor``
    - Stack inputs to inputs.
    - Convert inputs from bgr to rgb if the shape of input is (1, 3, H, W).
    - Normalize image with defined std and mean.
    - Do batch augmentations during training.
    - Record the information of ``batch_input_shape`` and ``pad_shape``.

    Args:
        mean (Sequence[Number], optional): The pixel mean of R, G, B channels.
            Defaults to None.
        std (Sequence[Number], optional): The pixel standard deviation of
            R, G, B channels. Defaults to None.
        pad_size_divisor (int): The size of padded image should be
            divisible by ``pad_size_divisor``. Defaults to 1.
        pad_value (Number): The padded pixel value. Defaults to 0.
        pad_mask (bool): Whether to pad instance masks. Defaults to False.
        mask_pad_value (int): The padded pixel value for instance masks.
            Defaults to 0.
        bgr_to_rgb (bool): whether to convert image from BGR to RGB.
            Defaults to False.
        rgb_to_bgr (bool): whether to convert image from RGB to RGB.
            Defaults to False.
        batch_augments (list[dict], optional): Batch-level augmentations
    """

    def __init__(self,
                 mean: Sequence[Number] = None,
                 std: Sequence[Number] = None,
                 pad_size_divisor: int = 1,
                 pad_value: Union[float, int] = 0,
                 pad_mask: bool = False,
                 mask_pad_value: int = 0,
                 bgr_to_rgb: bool = False,
                 rgb_to_bgr: bool = False,
                 batch_augments: Optional[List[dict]] = None):
        super().__init__()
        assert not (bgr_to_rgb and rgb_to_bgr), (
            '`bgr2rgb` and `rgb2bgr` cannot be set to True at the same time')
        assert (mean is None) == (std is None), (
            'mean and std should be both None or tuple')
        if mean is not None:
            assert len(mean) == 3 or len(mean) == 1, (
                'The length of mean should be 1 or 3 to be compatible with '
                f'RGB or gray image, but got {len(mean)}')
            assert len(std) == 3 or len(std) == 1, (  # type: ignore
                'The length of std should be 1 or 3 to be compatible with RGB '  # type: ignore # noqa: E501
                f'or gray image, but got {len(std)}')

            # Enable the normalization in preprocessing.
            self._enable_normalize = True
            self.register_buffer('mean',
                                 torch.tensor(mean).view(1, -1, 1, 1), False)
            self.register_buffer('std',
                                 torch.tensor(std).view(1, -1, 1, 1), False)
        else:
            self._enable_normalize = False

        self.channel_conversion = rgb_to_bgr or bgr_to_rgb
        self.pad_size_divisor = pad_size_divisor
        self.pad_value = pad_value
        self.pad_mask = pad_mask
        self.mask_pad_value = mask_pad_value
        if batch_augments is not None:
            self.batch_augments = nn.ModuleList(
                [MODELS.build(aug) for aug in batch_augments])
        else:
            self.batch_augments = None

    def forward(self, data: dict, training: bool = False) -> Dict:
        """Perform normalization、padding and bgr2rgb conversion based on
        ``TrackDataPreprocessor``.

        Args:
            data (dict): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            Tuple[Dict[str, List[torch.Tensor]], OptSampleList]: Data in the
            same format as the model input.
        """
        batch_pad_shape = self._get_pad_shape(data)
        data = super().forward(data=data, training=training)
        ori_inputs, data_samples = data['inputs'], data['data_samples']

        inputs = dict()
        for imgs_key, imgs in ori_inputs.items():
            # TODO: whether normalize should be after stack_batch
            # imgs is a list contain multiple Tensor of imgs.
            # The shape of imgs[0] is (T, C, H, W).
            channel = imgs[0].size(1)
            if self.channel_conversion and channel == 3:
                imgs = [_img[:, [2, 1, 0], ...] for _img in imgs]
            # change to `float`
            imgs = [_img.float() for _img in imgs]
            if self._enable_normalize:
                imgs = [(_img - self.mean) / self.std for _img in imgs]

            inputs[imgs_key] = stack_batch(imgs, self.pad_size_divisor,
                                           self.pad_value)

        if data_samples is not None:
            # NOTE the batched image size information may be useful, e.g.
            # in DETR, this is needed for the construction of masks, which is
            # then used for the transformer_head.
            for key, imgs in inputs.items():
                img_shape = tuple(imgs.size()[-2:])
                imgs_shape = [img_shape] * imgs.size(1) if imgs.size(
                    1) > 1 else img_shape
                ref_prefix = key[:-3]
                for data_sample, pad_shapes in zip(data_samples,
                                                   batch_pad_shape[key]):
                    data_sample.set_metainfo({
                        f'{ref_prefix}batch_input_shape':
                        imgs_shape,
                        f'{ref_prefix}pad_shape':
                        pad_shapes
                    })
                if self.pad_mask:
                    self.pad_gt_masks(data_samples, ref_prefix)

        if training and self.batch_augments is not None:
            for batch_aug in self.batch_augments:
                # Only yolox need batch_aug, and yolox can only process
                # `img` key. Therefore, only img is processed here.
                # The shape of `img` is (N, T, C, H, W), hence, we use
                # [:, 0] to change the shape to (N, C, H, W).
                assert len(inputs) == 1 and 'img' in inputs
                aug_inputs, data_samples = batch_aug(inputs['img'][:, 0],
                                                     data_samples)
                inputs['img'] = aug_inputs.unsqueeze(1)

        return dict(inputs=inputs, data_samples=data_samples)

    def _get_pad_shape(self, data: dict) -> Dict[str, List]:
        """Get the pad_shape of each image based on data and pad_size_divisor.

        Args:
            data (dict): Data sampled from dataloader.

        Returns:
            Dict[str, List]: The shape of padding.
        """
        batch_pad_shape = dict()
        for imgs_key in data['inputs']:
            pad_shape_list = []
            for imgs in data['inputs'][imgs_key]:
                pad_h = int(
                    np.ceil(imgs.shape[-2] /
                            self.pad_size_divisor)) * self.pad_size_divisor
                pad_w = int(
                    np.ceil(imgs.shape[-1] /
                            self.pad_size_divisor)) * self.pad_size_divisor
                pad_shapes = [
                    (pad_h, pad_w)
                ] * imgs.size(0) if imgs.size(0) > 1 else (pad_h, pad_w)
                pad_shape_list.append(pad_shapes)
            batch_pad_shape[imgs_key] = pad_shape_list
        return batch_pad_shape

    def pad_gt_masks(self, data_samples: Sequence[TrackDataSample],
                     ref_prefix: str) -> None:
        """Pad gt_masks to shape of batch_input_shape."""
        if 'masks' in data_samples[0].get(f'{ref_prefix}gt_instances'):
            for data_samples in data_samples:
                masks = data_samples.get(f'{ref_prefix}gt_instances').masks
                assert isinstance(masks, BitmapMasks)
                batch_input_shape = data_samples.get(
                    f'{ref_prefix}batch_input_shape')
                # handle cases where the number of image > 1
                if isinstance(batch_input_shape, list):
                    batch_input_shape = batch_input_shape[0]
                data_samples.get(
                    f'{ref_prefix}gt_instances').masks = masks.pad(
                        batch_input_shape, pad_val=self.mask_pad_value)
