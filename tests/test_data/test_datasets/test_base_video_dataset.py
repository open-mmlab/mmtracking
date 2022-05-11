# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from unittest import TestCase

from mmtrack.datasets import BaseVideoDataset

PREFIX = osp.join(osp.dirname(__file__), '../../data')
# This is a demo annotation file for CocoVideoDataset
# 1 videos, 2 categories ('car', 'person')
# 8 images, 2 instances -> [4, 3] objects
# 1 ignore, 2 crowd
DEMO_ANN_FILE = f'{PREFIX}/demo_cocovid_data/ann.json'


class TestBasevideoDataset(TestCase):

    @classmethod
    def setUpClass(cls):
        '''
        ann_file = ''
        metainfo = None
        data_root = None
        data_prefix = dict(img=None, ann=None)
        filter_cfg = None
        indices = None
        serialize_data = True
        pipeline = []
        test_mode = False
        lazy_init = False
        max_refetch = 1000
        '''

        cls.metainfo = dict(CLASSES=('car'))
        cls.ref_img_sampler = dict(
            num_ref_imgs=2,
            frame_range=4,
            filter_key_img=True,
            method='bilateral_uniform')
        cls.dataset = BaseVideoDataset(
            ann_file=DEMO_ANN_FILE,
            metainfo=cls.metainfo,
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            ref_img_sampler=cls.ref_img_sampler)

    def test_get_data_info(self):
        for i in range(len(self.dataset)):
            data_info = self.dataset.get_data_info(i)
            assert len(data_info['instances']) > 0

    def test_len(self):
        assert len(self.dataset) == 4

    def test_getitem(self):
        for i in range(len(self.dataset)):
            results = self.dataset[i]
            assert isinstance(results, dict)
            assert len(results['frame_id']) == 3
            assert abs(results['frame_id'][1] - results['frame_id'][0]
                       ) <= self.ref_img_sampler['frame_range']
            assert abs(results['frame_id'][2] - results['frame_id'][0]
                       ) <= self.ref_img_sampler['frame_range']
