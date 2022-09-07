# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence, Union

import cv2
import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def crop_image(image, crop_region, crop_size, padding=(0, 0, 0)):
    """Crop image based on `crop_region` and `crop_size`.

    Args:
        image (ndarray): of shape (H, W, 3).
        crop_region (ndarray): of shape (4, ) in [x1, y1, x2, y2] format.
        crop_size (int): Crop size.
        padding (tuple | ndarray): of shape (3, ) denoting the padding values.

    Returns:
        ndarray: Cropped image of shape (crop_size, crop_size, 3).
    """
    a = crop_size / (crop_region[2] - crop_region[0])
    b = crop_size / (crop_region[3] - crop_region[1])
    c = -a * crop_region[0]
    d = -b * crop_region[1]
    mapping = np.array([[a, 0, c], [0, b, d]]).astype(np.float32)
    crop_image = cv2.warpAffine(
        image,
        mapping, (crop_size, crop_size),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=padding)
    return crop_image


def imrenormalize(img: Union[Tensor, np.ndarray], img_norm_cfg: dict,
                  new_img_norm_cfg: dict) -> Union[Tensor, np.ndarray]:
    """Re-normalize the image.

    Args:
        img (Tensor | ndarray): Input image. If the input is a Tensor, the
            shape is (1, C, H, W). If the input is a ndarray, the shape
            is (H, W, C).
        img_norm_cfg (dict): Original configuration for the normalization.
        new_img_norm_cfg (dict): New configuration for the normalization.

    Returns:
        Tensor | ndarray: Output image with the same type and shape of
        the input.
    """
    if isinstance(img, torch.Tensor):
        assert img.ndim == 4 and img.shape[0] == 1
        new_img = img.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        new_img = _imrenormalize(new_img, img_norm_cfg, new_img_norm_cfg)
        new_img = new_img.transpose(2, 0, 1)[None]
        return torch.from_numpy(new_img).to(img)
    else:
        return _imrenormalize(img, img_norm_cfg, new_img_norm_cfg)


def _imrenormalize(img: Union[Tensor, np.ndarray], img_norm_cfg: dict,
                   new_img_norm_cfg: dict) -> Union[Tensor, np.ndarray]:
    """Re-normalize the image."""
    img_norm_cfg = img_norm_cfg.copy()
    new_img_norm_cfg = new_img_norm_cfg.copy()
    for k, v in img_norm_cfg.items():
        if (k == 'mean' or k == 'std') and not isinstance(v, np.ndarray):
            img_norm_cfg[k] = np.array(v, dtype=img.dtype)
    # reverse cfg
    if 'bgr_to_rgb' in img_norm_cfg:
        img_norm_cfg['rgb_to_bgr'] = img_norm_cfg['bgr_to_rgb']
        img_norm_cfg.pop('bgr_to_rgb')
    for k, v in new_img_norm_cfg.items():
        if (k == 'mean' or k == 'std') and not isinstance(v, np.ndarray):
            new_img_norm_cfg[k] = np.array(v, dtype=img.dtype)
    img = mmcv.imdenormalize(img, **img_norm_cfg)
    img = mmcv.imnormalize(img, **new_img_norm_cfg)
    return img


def gauss_blur(image: Tensor, kernel_size: Sequence,
               sigma: Sequence) -> Tensor:
    """The gauss blur transform.

    Args:
        image (Tensor): of shape (n, c, h, w)
        kernel_size (Tensor): The argument kernel size for gauss blur.
        sigma (Sequence): The argument sigma for gauss blur.

    Returns:
        Tensor: The blurred image.
    """
    assert len(kernel_size) == len(sigma) == 2
    x_coord = [
        torch.arange(-size, size + 1, dtype=torch.float32)
        for size in kernel_size
    ]
    filter = [
        torch.exp(-(x**2) / (2 * s**2)).to(image.device)
        for x, s in zip(x_coord, sigma)
    ]
    filter[0] = filter[0].view(1, 1, -1, 1) / filter[0].sum()
    filter[1] = filter[1].view(1, 1, 1, -1) / filter[1].sum()

    size = image.shape[2:]
    img_1 = F.conv2d(
        image.view(-1, 1, size[0], size[1]),
        filter[0],
        padding=(kernel_size[0], 0))

    img_2 = F.conv2d(
        img_1, filter[1],
        padding=(0, kernel_size[1])).view(1, -1, size[0], size[1])

    return img_2
