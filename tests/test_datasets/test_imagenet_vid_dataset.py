# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from unittest import TestCase

from mmtrack.datasets import ImagenetVIDDataset

PREFIX = osp.join(osp.dirname(__file__), '../data')
# This is a demo annotation file for CocoDataset
# 1 videos, 2 categories ('bus', 'car')
# 3 images, 6 instances
DEMO_ANN_FILE_IMG = f'{PREFIX}/demo_imagenetvid_data/ann_img.json'
DEMO_ANN_FILE_VID = f'{PREFIX}/demo_imagenetvid_data/ann_vid.json'


class TestImagenetVIDDataset(TestCase):

    @classmethod
    def setUpClass(cls):

        cls.metainfo = dict(classes=('bus', 'car'))
        cls.ref_img_sampler = dict(
            num_ref_imgs=2,
            frame_range=4,
            filter_key_img=False,
            method='bilateral_uniform')
        cls.dataset_video = ImagenetVIDDataset(
            ann_file=DEMO_ANN_FILE_VID,
            metainfo=cls.metainfo,
            load_as_video=True,
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            ref_img_sampler=cls.ref_img_sampler)
        cls.dataset_image = ImagenetVIDDataset(
            ann_file=DEMO_ANN_FILE_IMG,
            metainfo=cls.metainfo,
            load_as_video=False,
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            ref_img_sampler=cls.ref_img_sampler)

    def test_load_data_list(self):
        data_list, valid_data_indices = self.dataset_image.load_data_list()
        assert len(data_list) == 3
        assert valid_data_indices == [0, 1, 2]
        assert len(self.dataset_image) == 2

        data_list, valid_data_indices = self.dataset_video.load_data_list()
        assert len(data_list) == 8
        assert valid_data_indices == [0, 1, 2, 3, 4, 5, 6, 7]
        assert len(self.dataset_video) == 7
