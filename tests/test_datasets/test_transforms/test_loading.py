# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp

import numpy as np

from mmtrack.datasets.transforms import LoadTrackAnnotations


class TestLoadTrackAnnotations:

    def setup_class(cls):
        data_prefix = osp.join(osp.dirname(__file__), '../data')
        seg_map = osp.join(data_prefix, 'grayscale.jpg')
        cls.results = {
            'seg_map_path':
            seg_map,
            'instances': [{
                'bbox': [0, 0, 10, 20],
                'bbox_label': 1,
                'instance_id': 100,
                'keypoints': [1, 2, 3]
            }, {
                'bbox': [10, 10, 110, 120],
                'bbox_label': 2,
                'instance_id': 102,
                'keypoints': [4, 5, 6]
            }]
        }

    def test_load_instances_id(self):
        transform = LoadTrackAnnotations(
            with_bbox=False,
            with_label=True,
            with_instance_id=True,
            with_seg=False,
            with_keypoints=False,
        )
        results = transform(copy.deepcopy(self.results))
        assert 'gt_instances_id' in results
        assert (results['gt_instances_id'] == np.array([100, 102])).all()
        assert results['gt_instances_id'].dtype == np.int32

    def test_repr(self):
        transform = LoadTrackAnnotations(
            with_bbox=True,
            with_label=False,
            with_instance_id=True,
            with_seg=False,
            with_mask=False)
        assert repr(transform) == ('LoadTrackAnnotations(with_bbox=True, '
                                   'with_label=False, with_instance_id=True, '
                                   'with_mask=False, with_seg=False, '
                                   "poly2mask=True, imdecode_backend='cv2', "
                                   'file_client_args=None)')
