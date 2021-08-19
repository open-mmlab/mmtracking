# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmtrack.models.motion import LinearMotion


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
