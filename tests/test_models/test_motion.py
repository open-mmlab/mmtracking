import numpy as np
import torch
from mmdet.core.bbox.demodata import random_boxes

from mmtrack.models.motion import (CameraMotionCompensation, FlowNetSimple,
                                   LinearMotion)


def test_flownet_simple():
    # Test flownet_simple forward
    model = FlowNetSimple(img_scale_factor=0.5)
    model.init_weights()
    model.train()

    imgs = torch.randn(2, 6, 224, 224)
    img_metas = [
        dict(
            img_norm_cfg=dict(
                mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375)),
            img_shape=(224, 224, 3))
    ]
    flow = model(imgs, img_metas)
    assert flow.shape == torch.Size([2, 2, 224, 224])


def test_cmc():
    cmc = CameraMotionCompensation()
    img = np.random.randn(256, 256, 3).astype(np.float32)
    ref_img = img

    warp_matrix = cmc.get_warp_matrix(img, ref_img)
    assert isinstance(warp_matrix, torch.Tensor)

    bboxes = random_boxes(5, 256)
    trans_bboxes = cmc.warp_bboxes(bboxes, warp_matrix)
    assert (bboxes == trans_bboxes).all()


def test_linear_motion():
    linear_motion = LinearMotion(num_samples=2, center_motion=False)
    bboxes = [[1, 1, 1, 1], [3, 3, 3, 3], [6, 6, 6, 6]]
    bboxes = [torch.tensor(_, dtype=torch.float32) for _ in bboxes]
    bbox = linear_motion.step(bboxes)
    assert (bbox == torch.tensor([9., 9., 9., 9.])).all()

    linear_motion = LinearMotion(num_samples=3, center_motion=False)
    bboxes = [[1, 1, 1, 1], [3, 3, 3, 3], [6, 6, 6, 6]]
    bboxes = [torch.tensor(_, dtype=torch.float32) for _ in bboxes]
    bbox = linear_motion.step(bboxes)
    assert (bbox == torch.tensor([8.5, 8.5, 8.5, 8.5])).all()

    linear_motion = LinearMotion(num_samples=4, center_motion=False)
    bboxes = [[1, 1, 1, 1], [3, 3, 3, 3], [6, 6, 6, 6]]
    bboxes = [torch.tensor(_, dtype=torch.float32) for _ in bboxes]
    bbox = linear_motion.step(bboxes)
    assert (bbox == torch.tensor([8.5, 8.5, 8.5, 8.5])).all()

    linear_motion = LinearMotion(num_samples=4, center_motion=True)
    bboxes = [[1, 1, 1, 1], [3, 3, 3, 3], [6, 6, 6, 6]]
    bboxes = [torch.tensor(_, dtype=torch.float32) for _ in bboxes]
    bbox = linear_motion.step(bboxes)
    assert (bbox == torch.tensor([8.5, 8.5, 8.5, 8.5])).all()
