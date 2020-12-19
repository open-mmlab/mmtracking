import cv2
import numpy as np


def crop_image(image, crop_region, crop_size, padding=(0, 0, 0)):
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
