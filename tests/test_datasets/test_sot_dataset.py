# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from unittest import TestCase

from mmtrack.datasets import LaSOTDataset

PREFIX = osp.join(osp.dirname(__file__), '../data')
SOT_DATA_PREFIX = f'{PREFIX}/demo_sot_data'


class TestLaSOTDataset(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataset = LaSOTDataset(
            data_root=SOT_DATA_PREFIX,
            ann_file='trackingnet/annotations/trackingnet_train_infos.txt',
            data_prefix=dict(img='trackingnet'),
            test_mode=False)

    def test_get_bboxes_from_video(self):
        for idx in range(len(self.dataset)):
            bboxes = self.dataset.get_bboxes_from_video(idx)
            assert bboxes.shape[0] == self.dataset.get_len_per_video(idx)
            assert bboxes.shape[1] == 4

    def test_get_visibility_from_video(self):
        for idx in range(len(self.dataset)):
            visibility = self.dataset.get_visibility_from_video(idx)
            assert len(
                visibility['visible']) == self.dataset.get_len_per_video(idx)

    def test_get_infos_from_video(self):
        for idx in range(len(self.dataset)):
            video_info = self.dataset.get_infos_from_video(idx)
            assert len(video_info['frame_ids']) == 2
            assert len(video_info['bboxes']) == 2

    def test_prepare_test_data(self):
        for video_idx in range(len(self.dataset)):
            for frame_idx in range(self.dataset.get_len_per_video(video_idx)):
                test_data = self.dataset.prepare_test_data(
                    video_idx, frame_idx)
                assert len(test_data['instances']) > 0

    def test_prepare_train_data(self):
        for idx in range(len(self.dataset)):
            train_data = self.dataset.prepare_train_data(idx)
            assert len(train_data) == 2

    def test_prepare_data(self):
        for idx in range(len(self.dataset)):
            self.dataset.prepare_data(idx)

    def test_get_len_per_video(self):
        for idx in range(len(self.dataset)):
            assert self.dataset.get_len_per_video(idx) == 2

    def test_len(self):
        self.dataset.test_mode = False
        assert len(self.dataset) == 2
        self.dataset.test_mode = True
        assert len(self.dataset) == 4
        self.dataset.test_mode = False
