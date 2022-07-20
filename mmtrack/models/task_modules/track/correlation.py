# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn.functional as F


def depthwise_correlation(x, kernel):
    """Depthwise cross correlation.

    This function is proposed in
    `SiamRPN++ <https://arxiv.org/abs/1812.11703>`_.

    Args:
        x (Tensor): of shape (N, C, H_x, W_x).
        kernel (Tensor): of shape (N, C, H_k, W_k).

    Returns:
        Tensor: of shape (N, C, H_o, W_o). H_o = H_x - H_k + 1. So does W_o.
    """
    batch = kernel.size(0)
    channel = kernel.size(1)
    x = x.view(1, batch * channel, x.size(2), x.size(3))
    kernel = kernel.view(batch * channel, 1, kernel.size(2), kernel.size(3))
    out = F.conv2d(x, kernel, groups=batch * channel)
    out = out.view(batch, channel, out.size(2), out.size(3))
    return out
