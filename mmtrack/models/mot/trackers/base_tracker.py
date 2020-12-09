from abc import ABCMeta, abstractmethod

import torch
from addict import Dict

from mmtrack.models import TRACKERS


@TRACKERS.register_module()
class BaseTracker(metaclass=ABCMeta):

    def __init__(self, momentums=None, num_frames_retain=10):
        super().__init__()
        if momentums is not None:
            assert isinstance(momentums, dict), 'momentums must be a dict'
        self.momentums = momentums
        self.num_frames_retain

        self.reset()

    def reset(self):
        self.num_tracks = 0
        self.tracks = dict()

    @property
    def empty(self):
        return False if self.tracks else True

    @property
    def ids(self):
        return int(self.tracks.keys())

    def update(self, **kwargs):
        memo_items = list(kwargs.keys())
        if not hasattr(self, 'items'):
            self.memo_items = memo_items
        else:
            assert memo_items == self.memo_items

        assert 'ids' in memo_items
        num_objs = len(kwargs['ids'])
        id_indice = memo_items.index('ids')
        assert 'frame_ids' in memo_items
        if isinstance(kwargs['frame_ids'], int):
            kwargs['frame_ids'] = torch.tensor([kwargs['frame_ids']] *
                                               num_objs)
        cur_frame_id = int(kwargs['frame_ids'][0])
        for k, v in kwargs.items():
            if len(v) != num_objs:
                raise ValueError()

        for obj in zip(*kwargs.values()):
            id = int(obj[id_indice])
            if id in self.tracks:
                self.update_track(id, obj)
            else:
                self.init_track(id, obj)

        invalid_ids = []
        for k, v in self.tracks.items():
            if cur_frame_id - v['frame_ids'][-1] >= self.num_frames_retain:
                invalid_ids.append(k)
        for invalid_id in invalid_ids:
            self.tracks.pop(invalid_id)

    def update_track(self, id, obj):
        for k, v in zip(self.memo_items, obj):
            v = v[None]
            if k in self.momentums:
                m = self.momentums[k]
                self.tracks[id][k] = (1 - m) * self.self.tracks[id][k] + m * v
            else:
                self.tracks[id][k].append(v)

    def init_track(self, id, obj):
        self.tracks[id] = Dict()
        for k, v in zip(self.memo_items, obj):
            v = v[None]
            if k in self.momentums:
                self.tracks[id][k] = v
            else:
                self.tracks[id][k] = [v]

    @property
    def memo(self):
        outs = Dict()
        for k in self.memo_items:
            outs[k] = []

        for id, objs in self.tracks.items():
            for k, v in zip(self.memo_items, objs):
                v = v[-1] if k not in self.momentums else v

        for k, v in outs.items():
            outs[k] = torch.cat(v, dim=0)
        return outs

    def get(self, item, ids=None, num_samples=None):
        if ids is None:
            ids = self.ids

        outs = []
        for id in ids:
            out = self.tracks[id][item]
            if num_samples is not None:
                assert isinstance(out, list)
                out = out[-num_samples:]
                out = torch.cat(out, dim=0).mean(dim=0, keepdim=True)
            outs.append(out)
        return outs

    @abstractmethod
    def track(self, *args, **kwargs):
        pass
