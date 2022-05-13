# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from unittest import TestCase

from mmtrack.datasets import ReIDDataset

PREFIX = osp.join(osp.dirname(__file__), '../data')
# This is a demo annotation file for ReIDDataset.
REID_ANN_FILE = f'{PREFIX}/demo_reid_data/mot17_reid/ann.txt'


class TestReIDDataset(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.num_ids = 8
        cls.ins_per_id = 4
        cls.dataset = ReIDDataset(
            pipeline=[],
            ann_file=REID_ANN_FILE,
        )
        cls.dataset_triplet = ReIDDataset(
            pipeline=[],
            triplet_sampler=dict(
                num_ids=cls.num_ids, ins_per_id=cls.ins_per_id),
            ann_file=REID_ANN_FILE,
        )

    def test_get_data_info(self):
        # id 0 has 21 objects
        img_id = 0
        data_list = [
            self.dataset.get_data_info(i) for i in range(len(self.dataset))
        ]
        assert len([
            data_info for data_info in data_list
            if data_info['gt_label'] == img_id
        ]) == 21
        # id 11 doesn't have objects
        img_id = 11
        assert len([
            data_info for data_info in data_list
            if data_info['gt_label'] == img_id
        ]) == 0

    def test_len(self):
        assert len(self.dataset) == 704
        assert len(self.dataset_triplet) == 704

    def test_getitem(self):
        for i in range(len(self.dataset)):
            results = self.dataset[i]
            assert isinstance(results, dict)  # no triplet -> dict
            assert 'img_info' in results
            assert 'gt_label' in results
        for i in range(len(self.dataset_triplet)):
            results = self.dataset_triplet[i]
            assert isinstance(results, list)  # triplet -> list
            assert len(results) == self.num_ids * self.ins_per_id
            assert 'img_info' in results[0]
            assert 'gt_label' in results[0]
            assert results[0].keys() == results[1].keys()
            for idx in range(len(results) - 1):
                if (idx + 1) % self.ins_per_id != 0:
                    assert results[idx]['gt_label'] == \
                           results[idx + 1]['gt_label']
