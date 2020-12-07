from abc import ABCMeta, abstractmethod

import torch
from addict import Dict

from mmtrack.models import TRACKERS


@TRACKERS.register_module()
class BaseTracker(metaclass=ABCMeta):

    def __init__(self, momentums=None):
        super().__init__()
        if momentums is not None:
            assert isinstance(momentums, dict), 'momentums must be a dict'
        self.momentums = momentums

        self.reset()

    def reset(self):
        self.num_tracks = 0
        self.tracks = dict()

    @property
    def empty(self):
        return False if self.tracks else True

    def update(self, **kwargs):
        memo_items = list(kwargs.keys())
        if not hasattr(self, 'items'):
            self.memo_items = memo_items
        else:
            assert memo_items == self.memo_items

        assert 'ids' in memo_items
        id_indice = memo_items.index('ids')
        num_objs = len(kwargs['ids'])
        for k, v in kwargs.items():
            if len(v) != num_objs:
                if len(v) == 1:
                    kwargs[k] = [v] * num_objs
                else:
                    raise ValueError()

        for obj in zip(*kwargs.values()):
            id = int(obj[id_indice])
            if id in self.tracks:
                self.update_track(id, obj)
            else:
                self.init_track(id, obj)

    def update_track(self, id, obj):
        for k, v in zip(self.memo_items, obj):
            if k in self.momentums:
                m = self.momentums[k]
                self.tracks[id][k] = (1 - m) * self.self.tracks[id][k] + m * v
            else:
                self.tracks[id][k].append(v)

    def init_track(self, id, obj):
        self.tracks[id] = Dict()
        for k, v in zip(self.memo_items, obj):
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
                v = v[None]

        for k, v in outs.items():
            outs[k] = torch.cat(v, dim=0)
        return outs

    @abstractmethod
    def track(self, *args, **kwargs):
        pass
