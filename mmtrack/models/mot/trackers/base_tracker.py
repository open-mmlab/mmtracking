from abc import ABCMeta, abstractmethod

import torch
import torch.nn.functional as F
from addict import Dict

from mmtrack.models import TRACKERS


@TRACKERS.register_module()
class BaseTracker(metaclass=ABCMeta):

    def __init__(self, momentums=None, num_frames_retain=10):
        super().__init__()
        if momentums is not None:
            assert isinstance(momentums, dict), 'momentums must be a dict'
        self.momentums = momentums
        self.num_frames_retain = num_frames_retain

        self.reset()

    def reset(self):
        self.num_tracks = 0
        self.tracks = dict()

    @property
    def empty(self):
        return False if self.tracks else True

    @property
    def ids(self):
        return list(self.tracks.keys())

    @property
    def with_reid(self):
        """bool: whether the framework has a reid model"""
        return hasattr(self, 'reid') and self.reid is not None

    def update(self, **kwargs):
        memo_items = [k for k, v in kwargs.items() if v is not None]
        rm_items = [k for k in kwargs.keys() if k not in memo_items]
        for item in rm_items:
            kwargs.pop(item)
        if not hasattr(self, 'memo_items'):
            self.memo_items = memo_items
        else:
            assert memo_items == self.memo_items

        assert 'ids' in memo_items
        num_objs = len(kwargs['ids'])
        id_indice = memo_items.index('ids')
        assert 'frame_ids' in memo_items
        frame_id = int(kwargs['frame_ids'])
        if isinstance(kwargs['frame_ids'], int):
            kwargs['frame_ids'] = torch.tensor([kwargs['frame_ids']] *
                                               num_objs)
        # cur_frame_id = int(kwargs['frame_ids'][0])
        for k, v in kwargs.items():
            if len(v) != num_objs:
                raise ValueError()

        for obj in zip(*kwargs.values()):
            id = int(obj[id_indice])
            if id in self.tracks:
                self.update_track(id, obj)
            else:
                self.init_track(id, obj)

        self.pop_invalid_tracks(frame_id)

    def pop_invalid_tracks(self, frame_id):
        invalid_ids = []
        for k, v in self.tracks.items():
            if frame_id - v['frame_ids'][-1] >= self.num_frames_retain:
                invalid_ids.append(k)
        for invalid_id in invalid_ids:
            self.tracks.pop(invalid_id)

    def update_track(self, id, obj):
        for k, v in zip(self.memo_items, obj):
            v = v[None]
            if self.momentums is not None and k in self.momentums:
                m = self.momentums[k]
                self.tracks[id][k] = (1 - m) * self.tracks[id][k] + m * v
            else:
                self.tracks[id][k].append(v)

    def init_track(self, id, obj):
        self.tracks[id] = Dict()
        for k, v in zip(self.memo_items, obj):
            v = v[None]
            if self.momentums is not None and k in self.momentums:
                self.tracks[id][k] = v
            else:
                self.tracks[id][k] = [v]

    @property
    def memo(self):
        outs = Dict()
        for k in self.memo_items:
            outs[k] = []

        for id, objs in self.tracks.items():
            for k, v in objs.items():
                if k not in outs:
                    continue
                if self.momentums is not None and k in self.momentums:
                    v = v
                else:
                    v = v[-1]
                outs[k].append(v)

        for k, v in outs.items():
            outs[k] = torch.cat(v, dim=0)
        return outs

    def get(self, item, ids=None, num_samples=None, behavior=None):
        if ids is None:
            ids = self.ids

        outs = []
        for id in ids:
            out = self.tracks[id][item]
            if isinstance(out, list):
                if num_samples is not None:
                    out = out[-num_samples:]
                    out = torch.cat(out, dim=0)
                    if behavior == 'mean':
                        out = out.mean(dim=0, keepdim=True)
                    elif behavior is None:
                        out = out[None]
                    else:
                        raise NotImplementedError()
                else:
                    out = out[-1]
            outs.append(out)
        return torch.cat(outs, dim=0)

    @abstractmethod
    def track(self, *args, **kwargs):
        pass

    def crop_imgs(self, img, img_metas, bboxes, rescale=False):
        h, w, _ = img_metas[0]['img_shape']
        img = img[:, :, :h, :w]
        if rescale:
            bboxes[:, :4] *= torch.tensor(img_metas[0]['scale_factor']).to(
                bboxes.device)
        bboxes[:, 0::2] = torch.clamp(bboxes[:, 0::2], min=0, max=w)
        bboxes[:, 1::2] = torch.clamp(bboxes[:, 1::2], min=0, max=h)

        crop_imgs = []
        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            if x2 == x1:
                x2 = x1 + 1
            if y2 == y1:
                y2 = y1 + 1
            crop_img = img[:, :, y1:y2, x1:x2]
            if self.reid.get('img_scale', False):
                crop_img = F.interpolate(
                    crop_img,
                    size=self.reid['img_scale'],
                    mode='bilinear',
                    align_corners=False)
            crop_imgs.append(crop_img)

        if len(crop_imgs) > 0:
            return torch.cat(crop_imgs, dim=0)
        else:
            return img.new_zeros((0, ))
