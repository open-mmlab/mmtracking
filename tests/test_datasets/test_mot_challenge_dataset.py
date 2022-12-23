# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from unittest import TestCase

from mmtrack.datasets import MOTChallengeDataset
from mmtrack.datasets.api_wrappers import CocoVID

PREFIX = osp.join(osp.dirname(__file__), '../data')
# This is a demo annotation file for MOTChallengeDataset
# 1 video, 2 categories ('pedestrian')
# 3 images, 3 instances
# 0 ignore, 1 crowd
DEMO_ANN_FILE = f'{PREFIX}/demo_mot_data/ann.json'


class TestMOTChallengeDataset(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.metainfo = dict(classes=('pedestrian'))
        cls.ref_img_sampler = dict(
            num_ref_imgs=1,
            frame_range=2,
            filter_key_img=True,
            method='uniform')
        cls.dataset = MOTChallengeDataset(
            ann_file=DEMO_ANN_FILE,
            metainfo=cls.metainfo,
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            ref_img_sampler=cls.ref_img_sampler)

    def test_parse_data_info(self):
        coco = CocoVID(self.dataset.ann_file)

        img_ids = coco.get_img_ids_from_vid(vidId=1)
        for img_id in img_ids:
            # load img info
            raw_img_info = coco.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id
            raw_img_info['video_length'] = len(img_ids)

            # load ann info
            ann_ids = coco.get_ann_ids(img_ids=[img_id], cat_ids=1)
            raw_ann_info = coco.load_anns(ann_ids)

            # get data_info
            parsed_data_info = self.dataset.parse_data_info(
                dict(raw_img_info=raw_img_info, raw_ann_info=raw_ann_info))
            if img_id == 1:
                assert len(parsed_data_info['instances']) == 3
                assert parsed_data_info['instances'][0]['instance_id'] == 0
            else:
                assert len(parsed_data_info['instances']) == 0
