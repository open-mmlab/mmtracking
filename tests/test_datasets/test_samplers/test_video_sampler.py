# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from collections.abc import Iterable
from unittest import TestCase

from mmtrack.datasets import BaseVideoDataset, VideoSampler

PREFIX = osp.join(osp.dirname(__file__), '../../data')
DEMO_ANN_FILE = f'{PREFIX}/demo_cocovid_data/ann.json'


class TestBasevideoDataset(TestCase):

    @classmethod
    def setUpClass(cls):

        cls.metainfo = dict(CLASSES=('car'))
        cls.video_dataset = BaseVideoDataset(
            ann_file=DEMO_ANN_FILE,
            metainfo=cls.metainfo,
            ref_img_sampler=None,
            test_mode=True)
        cls.video_sampler = VideoSampler(cls.video_dataset)

    def test_iter(self):
        iterator = iter(self.video_sampler)
        assert isinstance(iterator, Iterable)
        for i in iterator:
            assert i >= 0 and i < len(self.video_dataset)

    def test_len(self):
        assert len(self.video_sampler) == len(self.video_dataset)
