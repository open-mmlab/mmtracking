# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from unittest import TestCase

from mmtrack.datasets import BaseVideoDataset

PREFIX = osp.join(osp.dirname(__file__), '../data')
# This is a demo annotation file for CocoVideoDataset
# 1 videos, 2 categories ('car', 'person')
# 8 images, 2 instances -> [4, 3] objects
# 1 ignore, 2 crowd
DEMO_ANN_FILE_VID = f'{PREFIX}/demo_cocovid_data/ann_vid.json'
DEMO_ANN_FILE_IMG = f'{PREFIX}/demo_cocovid_data/ann_img.json'


class TestBasevideoDataset(TestCase):

    @classmethod
    def setUpClass(cls):

        cls.metainfo = dict(classes=('car', ))
        cls.ref_img_sampler = dict(
            num_ref_imgs=2,
            frame_range=4,
            filter_key_img=True,
            method='bilateral_uniform')
        cls.dataset_video = BaseVideoDataset(
            ann_file=DEMO_ANN_FILE_VID,
            metainfo=cls.metainfo,
            load_as_video=True,
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            ref_img_sampler=cls.ref_img_sampler)
        cls.dataset_image = BaseVideoDataset(
            ann_file=DEMO_ANN_FILE_IMG,
            metainfo=cls.metainfo,
            load_as_video=False,
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            ref_img_sampler=cls.ref_img_sampler)

    def test_get_data_info(self):
        # test load_as_video=True
        for i in range(len(self.dataset_video)):
            data_info = self.dataset_video.get_data_info(i)
            assert len(data_info['instances']) > 0

        # test load_as_video=False
        for i in range(len(self.dataset_image)):
            data_info = self.dataset_image.get_data_info(i)
            assert len(data_info['instances']) > 0

    def test_len(self):
        assert len(self.dataset_video) == 5
        assert len(self.dataset_image) == 5

    def test_getitem(self):
        # test load_as_video=True
        for i in range(1, len(self.dataset_video) - 1):
            results = self.dataset_video[i]
            assert isinstance(results, dict)
            assert len(results['frame_id']) == 3
            assert abs(results['frame_id'][1] - results['frame_id'][0]
                       ) <= self.ref_img_sampler['frame_range']
            assert abs(results['frame_id'][2] - results['frame_id'][0]
                       ) <= self.ref_img_sampler['frame_range']

        # test load_as_video=False
        for i in range(1, len(self.dataset_image) - 1):
            results = self.dataset_image[i]
            assert isinstance(results, dict)
            assert len(results['img_id']) == 3
            assert len(set(results['img_id'])) == 1, \
                'all `img_id`s in the same item must be the same.'
