# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np


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
