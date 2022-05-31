# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from collections.abc import Iterable
from unittest import TestCase

from mmtrack.datasets import LaSOTDataset, QuotaSampler

PREFIX = osp.join(osp.dirname(__file__), '../../data')
SOT_DATA_PREFIX = f'{PREFIX}/demo_sot_data'


class TestBasevideoDataset(TestCase):

    @classmethod
    def setUpClass(cls):

        cls.video_dataset = LaSOTDataset(
            data_root=SOT_DATA_PREFIX,
            ann_file='trackingnet/annotations/trackingnet_train_infos.txt',
            data_prefix=dict(img_path='trackingnet'),
            test_mode=False)
        cls.video_sampler = QuotaSampler(
            cls.video_dataset, samples_per_epoch=10)

    def test_iter(self):
        iterator = iter(self.video_sampler)
        assert isinstance(iterator, Iterable)
        for i in iterator:
            assert i >= 0 and i < len(self.video_dataset)

    def test_len(self):
        assert len(self.video_sampler) == 10
