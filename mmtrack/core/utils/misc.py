# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from mmtrack.core import TrackDataSample


def stack_batch(tensors: List[torch.Tensor],
                pad_size_divisor: int = 0,
                pad_value: Union[int, float] = 0) -> torch.Tensor:
    """Stack multiple tensors to form a batch and pad the images to the max
    shape use the right bottom padding mode in these images. If
    ``pad_size_divisor > 0``, add padding to ensure the common height and width
    is divisible by ``pad_size_divisor``.

    Args:
        tensors (List[Tensor]): The input multiple tensors. each is a
            TCHW 4D-tensor. T denotes the number of key/reference frames.
        pad_size_divisor (int): If ``pad_size_divisor > 0``, add padding
            to ensure the common height and width is divisible by
            ``pad_size_divisor``. This depends on the model, and many
            models need a divisibility of 32. Defaults to 0
        pad_value (int, float): The padding value. Defaults to 0

    Returns:
       Tensor: The NTCHW 5D-tensor. N denotes the batch size.
    """
    assert isinstance(tensors, list), \
        f'Expected input type to be list, but got {type(tensors)}'
    assert len(set([tensor.ndim for tensor in tensors])) == 1, \
        f'Expected the dimensions of all tensors must be the same, ' \
        f'but got {[tensor.ndim for tensor in tensors]}'
    assert tensors[0].ndim == 4, f'Expected tensor dimension to be 4, ' \
                                 f'but got {tensors[0].ndim}'
    assert len(set([tensor.shape[0] for tensor in tensors])) == 1, \
        f'Expected the channels of all tensors must be the same, ' \
        f'but got {[tensor.shape[0] for tensor in tensors]}'

    tensor_sizes = [(tensor.shape[-2], tensor.shape[-1]) for tensor in tensors]
    max_size = np.stack(tensor_sizes).max(0)

    if pad_size_divisor > 1:
        # the last two dims are H,W, both subject to divisibility requirement
        max_size = (
            max_size +
            (pad_size_divisor - 1)) // pad_size_divisor * pad_size_divisor

    padded_samples = []
    for tensor in tensors:
        padding_size = [
            0, max_size[-1] - tensor.shape[-1], 0,
            max_size[-2] - tensor.shape[-2]
        ]
        if sum(padding_size) == 0:
            padded_samples.append(tensor)
        else:
            padded_samples.append(F.pad(tensor, padding_size, value=pad_value))

    return torch.stack(padded_samples, dim=0)


def convert_data_sample_type(
        data_sample: TrackDataSample,
        num_ref_imgs: int = 1) -> Tuple[List[TrackDataSample], List[dict]]:
    """Convert the type of ``data_sample`` from dict[list] to list[dict].

    Note: This function is mainly used to be compatible with the
        interface of MMDetection. It make sure that the information of
        each reference image can be independently packed into
        ``data_sample`` in which all the keys are without prefix "ref_".

    Args:
        data_sample (TrackDataSample): Data sample input.
        num_ref_imgs (int, optional): The numbe of reference images in the
            ``data_sample``. Defaults to 1.

    Returns:
        Tuple[List[TrackDataSample], List[dict]]: The first element is the
            list of object of TrackDataSample. The second element is the
            list of meta information of reference images.
    """
    ref_data_samples, ref_metainfos = [], []
    for _ in range(num_ref_imgs):
        ref_data_samples.append(deepcopy(data_sample))
        ref_metainfos.append(deepcopy(data_sample.metainfo))

    for key, value in data_sample.metainfo.items():
        if key.startswith('ref_'):
            new_key = key[4:]
            if num_ref_imgs == 1:
                value = [value]
            assert len(value) == num_ref_imgs
            for i, v in enumerate(value):
                ref_metainfos[i][new_key] = v
                ref_data_samples[i].set_metainfo(dict(new_key=v))
                # pop the redundant original reference key.
                ref_metainfos[i].pop(key)
                ref_data_samples[i].pop(key)

    return ref_data_samples, ref_metainfos


def max2d(input):
    """Computes the value and position of maximum in the last two dimensions.

    Args:
        input (Tensor): of shape (..., H, W)

    Returns:
        max_val (Tensor): The maximum value.
        argmax (Tensor): The position of maximum in [row, col] format.
    """

    max_val_row, argmax_row = torch.max(input, dim=-2)
    max_val, argmax_col = torch.max(max_val_row, dim=-1)
    argmax_row = argmax_row.view(argmax_col.numel(),
                                 -1)[torch.arange(argmax_col.numel()),
                                     argmax_col.view(-1)]
    argmax_row = argmax_row.reshape(argmax_col.shape)
    argmax = torch.cat((argmax_row.unsqueeze(-1), argmax_col.unsqueeze(-1)),
                       -1)
    return max_val, argmax
