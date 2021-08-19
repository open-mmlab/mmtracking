# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmdet.core.bbox.demodata import random_boxes

from mmtrack.models.motion import CameraMotionCompensation


def test_cmc():
    cmc = CameraMotionCompensation()
    img = np.random.randn(256, 256, 3).astype(np.float32)
    ref_img = img

    warp_matrix = cmc.get_warp_matrix(img, ref_img)
    assert isinstance(warp_matrix, torch.Tensor)

    bboxes = random_boxes(5, 256)
    trans_bboxes = cmc.warp_bboxes(bboxes, warp_matrix)
    assert (bboxes == trans_bboxes).all()
