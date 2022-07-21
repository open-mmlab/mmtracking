# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple, Union

import cv2
import mmcv
import numpy as np
import torch
from torch import Tensor

cv2_interp_codes = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4
}

cv2_border_modes = {
    'constant': cv2.BORDER_CONSTANT,
    'replicate': cv2.BORDER_REPLICATE,
    'reflect': cv2.BORDER_REFLECT,
    'wrap': cv2.BORDER_WRAP,
}


def ndarray2tensor(x, device='cpu'):
    assert isinstance(x, np.ndarray)
    return torch.from_numpy(x.transpose(2, 0,
                                        1)).float().unsqueeze(0).to(device)


def tensor2ndarray(x):
    assert isinstance(x, torch.Tensor)
    if x.is_cuda:
        x = x.cpu()
    return x.squeeze(0).permute(1, 2, 0).numpy()


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


def rotate_image(img: np.ndarray,
                 angle: float,
                 center: Optional[Tuple[float, float]] = None,
                 scale: float = 1.0,
                 border_mode: str = 'constant',
                 border_value: int = 0,
                 interpolation: str = 'bilinear',
                 auto_bound: bool = False) -> np.ndarray:
    """Rotate an image.

    Args:
        img (np.ndarray): of shape (H, W, 3).
        angle (float): Rotation angle in degrees, positive values mean
            clockwise rotation.
        center (tuple[float], optional): Center point (w, h) of the rotation in
            the source image. If not specified, the center of the image will be
            used.
        scale (float): Isotropic scale factor.
        border_mode (cv2.BorderTypes): Default to `cv2.BORDER_CONSTANT`
        border_value (int): Border value onlu used in `cv2.BORDER_CONSTANT`
            mode.
        interpolation (str): Default to `cv2.INTER_LINEAR`
        auto_bound (bool): Whether to adjust the image size to cover the whole
            rotated image.
    Returns:
        np.ndarray: The rotated image.
    """
    if center is not None and auto_bound:
        raise ValueError('`auto_bound` conflicts with `center`')
    h, w = img.shape[:2]
    if center is None:
        center = ((w - 1) * 0.5, (h - 1) * 0.5)
    assert isinstance(center, tuple)

    matrix = cv2.getRotationMatrix2D(center, -angle, scale)
    if auto_bound:
        cos = np.abs(matrix[0, 0])
        sin = np.abs(matrix[0, 1])
        new_w = h * sin + w * cos
        new_h = h * cos + w * sin
        matrix[0, 2] += (new_w - w) * 0.5
        matrix[1, 2] += (new_h - h) * 0.5
        w = int(np.round(new_w))
        h = int(np.round(new_h))
    rotated = cv2.warpAffine(
        img,
        matrix, (w, h),
        flags=cv2_interp_codes[interpolation],
        borderMode=cv2_border_modes[border_mode],
        borderValue=border_value)
    return rotated
