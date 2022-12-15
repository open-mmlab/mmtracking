# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from unittest import TestCase

from mmtrack.datasets import (BaseVideoDataset, EntireVideoBatchSampler,
                              VideoSampler)

PREFIX = osp.join(osp.dirname(__file__), '../../data')
DEMO_ANN_FILE = f'{PREFIX}/demo_cocovid_data/ann_vid.json'


class TestEntireVideoBatchSampler(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.metainfo = dict(classes=('car'))
        cls.video_dataset = BaseVideoDataset(
            ann_file=DEMO_ANN_FILE,
            metainfo=cls.metainfo,
            ref_img_sampler=None,
            test_mode=True)
        cls.video_sampler = VideoSampler(cls.video_dataset)

    def test_video_batch(self):
        batch_size = 1
        batch_sampler = EntireVideoBatchSampler(
            self.video_sampler, batch_size=batch_size)
        # 1 video
        self.assertEqual(len(batch_sampler), 1)
