# Copyright (c) OpenMMLab. All rights reserved.
# The codes are modified from https://github.com/visionml/pytracking/blob/master/ltr/models/layers/filter.py # noqa: E501
import torch.nn.functional as F
from torch import Tensor


def apply_filter(feat: Tensor, filter_weights: Tensor) -> Tensor:
    """Applies the filter on the input features.

    The number of groups is automatically calculated.
    args:
        feat (Tensor): The input features with two possible shapes in the
            different modes:
            - training mode: of shape (num_img_per_seq, bs, c, h, w)
            - test mode: of shape (num_img_per_seq, c, h, w).
        filter_weights (Tensor): The filter to be applied on the `feat`. There
            are two possible shapes in the different modes:
            - training mode: of shape (bs, c, filter_h, filter_w)
            - test mode: of shape (1, c, filter_h, filter_w)
    output:
        scores (Tenosr): Output of filtering.
            - train mode: (num_img_per_seq, bs, h, w)
            - test mode: (num_img_per_seq, 1, h, w)
    """
    padding = (filter_weights.shape[-2] // 2, filter_weights.shape[-1] // 2)
    num_groups = feat.shape[1] if feat.dim() == 5 else 1
    scores = F.conv2d(
        feat.reshape(feat.shape[0], -1, *feat.shape[-2:]),
        filter_weights,
        padding=padding,
        groups=num_groups)
    return scores


def apply_feat_transpose(feat: Tensor,
                         activation: Tensor,
                         filter_size_hw: Tensor,
                         training: bool = True) -> Tensor:
    """The transposed operation of `apply_filter` w.r.t the filter. It can be
    used to compute the filter gradient. There are two implements: the one
    forwards slowly and backwards fast, which used in training, and the other
    is the opposite, which used in test.

    Args:
        feat (Tensor): The input features with two possible shapes in the
            different modes:
            - training mode: of shape (num_img_per_seq, bs, c, h, w)
            - test mode: of shape (num_img_per_seq, c, h, w).
        activation (Tensor): The activation (e.g. residuals between output and
            label). There are two possible shapes in the different modes:
            - training mode: of shape (num_img_per_seq, bs, size_h, size_w)
            - test mode: of shape (num_img_per_seq, 1, size_h, size_w)
        filter_size_hw (Tensor): of shape (2,) shape in [h, w] format.
        training (bool, optional): Whether training mode or not. The faster
            implementation is chose according to this.

    Returns:
        Tensor. There are two possible shape in the
            different mode:
            - training mode: of shape (bs, c, filter_h, fiter_w).
            - test mode: of shape (1, c, filter_h, fiter_w).
    """

    if isinstance(filter_size_hw, int):
        filter_size_hw = (filter_size_hw, filter_size_hw)

    if training:
        # slow forward and fast backward
        # TODO: check the pad difference in training and test mode
        transpose_pad = [sz // 2 for sz in filter_size_hw]
        num_img_per_seq = feat.shape[0]
        batch_size = feat.shape[1] if feat.dim() == 5 else 1

        filter_grad = F.conv2d(
            feat.reshape(-1, *feat.shape[-3:]).permute(1, 0, 2, 3),
            activation.reshape(-1, 1, *activation.shape[-2:]),
            padding=transpose_pad,
            groups=num_img_per_seq * batch_size)

        # The shape of returns is (bs, c, filter_h, fiter_w)
        if num_img_per_seq == 1:
            return filter_grad.permute(1, 0, 2, 3)

        filter_grad = filter_grad.view(-1, num_img_per_seq, batch_size,
                                       *filter_grad.shape[-2:])
        return filter_grad.sum(dim=1).permute(1, 0, 2, 3)
    else:
        # fast forwward and slow backward
        transpose_pad = [(sz - 1) // 2 for sz in filter_size_hw]
        batch_size = feat.shape[0]
        filter_grad = F.conv2d(
            activation.reshape(1, -1, *activation.shape[-2:]),
            feat.reshape(-1, 1, *feat.shape[-2:]),
            padding=transpose_pad,
            groups=batch_size)

        filter_grad = filter_grad.view(batch_size, 1, -1,
                                       *filter_grad.shape[-2:])
        # The shape of returns is (1, c, filter_h, fiter_w)
        return filter_grad.sum(dim=0).flip((2, 3))
