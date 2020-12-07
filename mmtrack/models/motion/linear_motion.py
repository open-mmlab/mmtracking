import torch

from ..builder import MOTION


@MOTION.register_module()
class LinearMotion(object):

    def __init__(self, num_samples=2, center_motion=False):
        self.num_samples = num_samples
        self.center_motion = center_motion

    def center(self, bbox):
        x1, x2, y1, y2 = bbox
        return torch.Tensor([(x2 + x1) / 2, (y2 + y1) / 2]).to(bbox.device)

    def get_velocity(self, bboxes):
        num_bboxes = len(bboxes)
        num_samples = num_bboxes if self.num_samples > num_bboxes \
            else self.num_samples
        if self.center_motion:
            vs = [
                self.center(b2) - self.center(b1)
                for (b1,
                     b2) in zip(bboxes[-num_samples:], bboxes[-num_samples +
                                                              1:])
            ]
        else:
            vs = [
                b2 - b1
                for (b1,
                     b2) in zip(bboxes[-num_samples:], bboxes[-num_samples +
                                                              1:])
            ]
        return torch.stack(vs, dim=0).mean(dim=0)

    def step(self, bboxes):
        assert isinstance(bboxes, list)
        vs = self.get_velocity(bboxes)

        if self.center_motion:
            cx, cy = self.center(bboxes[-1]) + vs
            w = bboxes[-1][2] - bboxes[-1][0]
            h = bboxes[-1][3] - bboxes[-1][1]
            bbox = torch.Tensor(
                [cx - w / 2, cy - h / 2, cx + w / 2,
                 cy + h / 2]).to(bboxes[-1].device)
        else:
            bbox = bboxes[-1] + vs
        return bbox
