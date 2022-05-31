# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from unittest import TestCase

from mmtrack.registry import DATASETS
from mmtrack.utils import register_all_modules

PREFIX = osp.join(osp.dirname(__file__), '../data/demo_sot_data/')


class TestRandomSampleConcatDataset(TestCase):

    @classmethod
    def setUpClass(cls):
        register_all_modules(init_default_scope=True)
        train_cfg = dict(
            type='RandomSampleConcatDataset',
            dataset_sampling_weights=[1, 1],
            datasets=[
                dict(
                    type='GOT10kDataset',
                    data_root=PREFIX,
                    ann_file=  # noqa: E251
                    'trackingnet/annotations/trackingnet_train_infos.txt',  # noqa: E501
                    data_prefix=dict(img_path='trackingnet'),
                    pipeline=[],
                    test_mode=False),
                dict(
                    type='TrackingNetDataset',
                    chunks_list=[0],
                    data_root=PREFIX,
                    ann_file=  # noqa: E251
                    'trackingnet/annotations/trackingnet_train_infos.txt',  # noqa: E501
                    data_prefix=dict(img_path='trackingnet'),
                    pipeline=[],
                    test_mode=False)
            ])

        cls.dataset = DATASETS.build(train_cfg)

    def test_get_item(self):
        results = self.dataset[0]
        assert len(self.dataset) == 4
        assert self.dataset.dataset_sampling_probs == [0.5, 0.5]
        assert len(results) == 2
