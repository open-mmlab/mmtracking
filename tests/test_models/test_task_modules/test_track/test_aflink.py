# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np
from torch import nn

from mmtrack.registry import TASK_UTILS
from mmtrack.utils import register_all_modules


class TestAppearanceFreeLink(TestCase):

    @classmethod
    def setUpClass(cls):
        register_all_modules(init_default_scope=True)
        cls.cfg = dict(
            type='AppearanceFreeLink',
            checkpoint='',
            temporal_threshold=(0, 30),
            spatial_threshold=75,
            confidence_threshold=0.95,
        )

    def test_init(self):
        aflink = TASK_UTILS.build(self.cfg)
        assert aflink.temporal_threshold == (0, 30)
        assert aflink.spatial_threshold == 75
        assert aflink.confidence_threshold == 0.95
        assert isinstance(aflink.model, nn.Module)

    def test_forward(self):
        pred_track = np.random.randn(10, 7)
        aflink = TASK_UTILS.build(self.cfg)
        linked_track = aflink.forward(pred_track)
        assert isinstance(linked_track, np.ndarray)
        assert linked_track.shape == (10, 7)
