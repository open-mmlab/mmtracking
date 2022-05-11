# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp

import numpy as np

from mmtrack.datasets import PIPELINES


class TestLoading(object):

    @classmethod
    def setup_class(cls):
        cls.data_prefix = osp.join(osp.dirname(__file__), '../../data')

    def test_load_seq_imgs(self):
        img_names = ['image_1.jpg', 'image_2.jpg', 'image_3.jpg']
        results = [
            dict(img_prefix=self.data_prefix, img_info=dict(filename=name))
            for name in img_names
        ]
        load = PIPELINES.get('LoadMultiImagesFromFile')()
        all_results = load(copy.deepcopy(results))
        assert isinstance(all_results, list)
        for i, results in enumerate(all_results):
            assert results['filename'] == osp.join(self.data_prefix,
                                                   img_names[i])
            assert results['ori_filename'] == img_names[i]
            assert results['img'].shape == (256, 512, 3)
            assert results['img'].dtype == np.uint8
            assert results['img_shape'] == (256, 512, 3)
            assert results['ori_shape'] == (256, 512, 3)

    def test_load_detections(self):
        results = dict()
        results['bbox_fields'] = []
        results['detections'] = [np.random.randn(4, 5), np.random.randn(3, 5)]
        load = PIPELINES.get('LoadDetections')()
        results = load(results)
        assert 'public_bboxes' in results
        assert 'public_scores' in results
        assert 'public_labels' in results
        assert results['public_bboxes'].shape == (7, 4)
        assert results['public_scores'].shape == (7, )
        assert results['public_labels'].shape == (7, )
        assert 'public_bboxes' in results['bbox_fields']
