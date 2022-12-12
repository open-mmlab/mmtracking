# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from collections.abc import Iterable
from unittest import TestCase

from mmtrack.datasets import BaseVideoDataset, LaSOTDataset, VideoSampler

PREFIX = osp.join(osp.dirname(__file__), '../../data')
DEMO_ANN_FILE = f'{PREFIX}/demo_cocovid_data/ann_vid.json'
SOT_DATA_PREFIX = f'{PREFIX}/demo_sot_data'


class TestBasevideoDataset(TestCase):

    @classmethod
    def setUpClass(cls):

        cls.metainfo = dict(classes=('car'))
        cls.video_dataset = BaseVideoDataset(
            ann_file=DEMO_ANN_FILE,
            metainfo=cls.metainfo,
            ref_img_sampler=None,
            test_mode=True)
        cls.video_sampler = VideoSampler(cls.video_dataset)

        cls.sot_video_dataset = LaSOTDataset(
            data_root=SOT_DATA_PREFIX,
            ann_file='trackingnet/annotations/trackingnet_train_infos.txt',
            data_prefix=dict(img_path='trackingnet'),
            test_mode=True)
        cls.sot_video_sampler = VideoSampler(cls.sot_video_dataset)

    def test_iter(self):
        iterator = iter(self.video_sampler)
        assert isinstance(iterator, Iterable)
        for i in iterator:
            assert i >= 0 and i < len(self.video_dataset)

        iterator = iter(self.sot_video_sampler)
        assert isinstance(iterator, Iterable)
        for idx in iterator:
            assert len(idx) == 2
            video_idx, frame_idx = idx
            assert (video_idx >= 0
                    and video_idx < self.sot_video_dataset.num_videos)
            assert (frame_idx >= 0 and frame_idx <
                    self.sot_video_dataset.get_len_per_video(video_idx))

    def test_len(self):
        assert len(self.video_sampler) == len(self.video_dataset)
        assert len(self.sot_video_sampler) == len(self.sot_video_dataset)
