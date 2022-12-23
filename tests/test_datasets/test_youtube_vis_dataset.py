# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from unittest import TestCase

from mmtrack.datasets import YouTubeVISDataset

PREFIX = osp.join(osp.dirname(__file__), '../data')
# This is a demo annotation file for YouTubeVISDataset
# 1 video, 1 categories ('sedan')
# 1 images, 1 instances
# 0 crowd
DEMO_ANN_FILE = f'{PREFIX}/demo_vis_data/ann.json'


class TestYouTubeVISDataset(TestCase):

    @classmethod
    def setUpClass(cls):

        cls.dataset = YouTubeVISDataset(
            ann_file=DEMO_ANN_FILE, dataset_version='2019')

    def test_set_dataset_classes(self):
        assert isinstance(self.dataset.metainfo, dict)
        assert len(self.dataset.metainfo['classes']) == 40
