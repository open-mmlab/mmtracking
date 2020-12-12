import torch

from ..builder import MOTION


@MOTION.register_module()
class LinearMotion(object):

    def __init__(self, num_samples=2, center_motion=False):
        self.num_samples = num_samples
        self.center_motion = center_motion

    def center(self, bbox):
        if bbox.ndim == 2:
            assert bbox.shape[0] == 1
            bbox = bbox[0]
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

    def step(self, bboxes, velocity=None):
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
        for k, v in tracks.items():
            if int(v.frame_ids[-1]) == frame_id - 1:
                v.velocity = self.get_velocity(v.bboxes)
            if hasattr(v, 'velocity'):
                v.bboxes[-1] = self.step(v.bboxes, v.velocity)[None]
        return tracks
