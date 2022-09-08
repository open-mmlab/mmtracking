# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmtrack.utils import gauss_blur


def test_gauss_blur():
    img = torch.randn(1, 2, 10, 10)
    blurred_img = gauss_blur(img, kernel_size=(5, 5), sigma=(2, 2))
    assert blurred_img.shape == img.shape
