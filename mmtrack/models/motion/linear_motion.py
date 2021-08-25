# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..builder import MOTION


@MOTION.register_module()
class LinearMotion(object):
    """Linear motion while tracking.

    Args:
        num_samples (int, optional): Number of samples to calculate the
            velocity. Default to 2.
        center_motion (bool, optional): Whether use center location or
            bounding box location to estimate the velocity. Default to False.
    """

    def __init__(self, num_samples=2, center_motion=False):
        self.num_samples = num_samples
        self.center_motion = center_motion

    def center(self, bbox):
        """Get the center of the box."""
        if bbox.ndim == 2:
            assert bbox.shape[0] == 1
            bbox = bbox[0]
        x1, y1, x2, y2 = bbox
        return torch.Tensor([(x2 + x1) / 2, (y2 + y1) / 2]).to(bbox.device)

    def get_velocity(self, bboxes, num_samples=None):
        """Get velocities of the input objects."""
        if num_samples is None:
            num_samples = min(len(bboxes), self.num_samples)

        vs = []
        for (b1, b2) in zip(bboxes[-num_samples:], bboxes[-num_samples + 1:]):
            if self.center_motion:
                v = self.center(b2) - self.center(b1)
            else:
                v = b2 - b1
            vs.append(v)
        return torch.stack(vs, dim=0).mean(dim=0)

    def step(self, bboxes, velocity=None):
        """Step forward with the velocity."""
        assert isinstance(bboxes, list)
        if velocity is None:
            velocity = self.get_velocity(bboxes)
        bbox = bboxes[-1]
        if bbox.ndim == 2:
            assert bbox.shape[0] == 1
            bbox = bbox[0]

        if self.center_motion:
            cx, cy = self.center(bbox) + velocity
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            bbox = torch.Tensor(
                [cx - w / 2, cy - h / 2, cx + w / 2,
                 cy + h / 2]).to(bbox.device)
        else:
            bbox += velocity
        return bbox

    def track(self, tracks, frame_id):
        """Tracking forward."""
        for k, v in tracks.items():
            if int(v.frame_ids[-1]) == frame_id - 1:
                rids = v.frame_ids[::-1]
                num_bboxes = len(v.bboxes)
                for n, (i, j) in enumerate(zip(rids, rids[1:]), 1):
                    if i != j + 1:
                        num_bboxes = n
                num_samples = min(num_bboxes, self.num_samples)
                v.velocity = self.get_velocity(v.bboxes, num_samples)
            if 'velocity' in v:
                v.bboxes[-1] = self.step(v.bboxes, v.velocity)[None]
        return tracks
