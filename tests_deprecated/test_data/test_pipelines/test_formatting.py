# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp

import numpy as np
import pytest
from mmcv.utils import build_from_cfg

from mmtrack.datasets import PIPELINES


class TestFormatting(object):

    @classmethod
    def setup_class(cls):
        cls.data_prefix = osp.join(osp.dirname(__file__), '../../data')

    def test_formatting(self):
        img_names = ['image_1.jpg', 'image_2.jpg', 'image_3.jpg']
        collect_keys = ['img', 'gt_bboxes', 'gt_label']
        num_objects = 4
        num_ref_imgs = len(img_names) - 1

        results = [
            dict(img_prefix=self.data_prefix, img_info=dict(filename=name))
            for name in img_names
        ]

        load = dict(type='LoadMultiImagesFromFile')
        load = build_from_cfg(load, PIPELINES)
        results = load(results)
        assert len(results) == len(img_names)

        for _result in results:
            _result['padding_mask'] = np.ones_like(_result['img'], dtype=bool)
        check_data_validity = dict(type='CheckPadMaskValidity', stride=16)
        check_data_validity = build_from_cfg(check_data_validity, PIPELINES)
        assert results is not None

        for result in results:
            result['gt_bboxes'] = np.random.randn(num_objects, 4)
            result['gt_label'] = np.random.randint(0, 10)

        collect = dict(type='VideoCollect', keys=collect_keys)
        collect = build_from_cfg(collect, PIPELINES)
        results = collect(results)
        assert len(results) == len(img_names)
        for key in collect_keys:
            assert key in results[0]
            assert key in results[1]
            assert key in results[2]
        assert 'img_metas' in results[0]
        assert 'img_metas' in results[1]
        assert 'img_metas' in results[2]
        key_results = results[0]

        # the type of results is a list
        # the length of results is greater than 1
        reid_results = copy.deepcopy(results)
        bundle = dict(type='ReIDFormatBundle')
        bundle = build_from_cfg(bundle, PIPELINES)
        reid_results = bundle(reid_results)
        assert isinstance(reid_results, dict)
        assert 'img' in reid_results
        assert not reid_results['img'].cpu_only
        assert reid_results['img'].stack
        assert reid_results['img'].data.ndim == 4
        assert reid_results['img'].data.size(0) == 3
        assert 'gt_label' in reid_results
        assert not reid_results['gt_label'].cpu_only
        assert reid_results['gt_label'].stack
        assert reid_results['gt_label'].data.ndim == 1
        assert reid_results['img'].data.size(0) == 3

        # the type of results is a dict
        reid_results = copy.deepcopy(results[0])
        reid_results = bundle(reid_results)
        assert isinstance(reid_results, dict)
        assert 'img' in reid_results
        assert not reid_results['img'].cpu_only
        assert reid_results['img'].stack
        assert reid_results['img'].data.ndim == 3
        assert 'gt_label' in reid_results
        assert not reid_results['gt_label'].cpu_only
        assert reid_results['gt_label'].stack
        assert reid_results['gt_label'].data.ndim == 1

        # the type of results is a tuple
        with pytest.raises(TypeError):
            reid_results = (copy.deepcopy(results[0]), )
            reid_results = bundle(reid_results)

        # the type of results is a list but it only has one item
        with pytest.raises(AssertionError):
            reid_results = [copy.deepcopy(results[0])]
            reid_results = bundle(reid_results)

        concat2twoparts = dict(type='ConcatSameTypeFrames', num_key_frames=2)
        concat2twoparts = build_from_cfg(concat2twoparts, PIPELINES)
        concat_video_results = concat2twoparts(copy.deepcopy(results))
        assert len(concat_video_results) == 2
        assert concat_video_results[0]['img'].ndim == 4
        assert concat_video_results[0]['img'].shape[3] == 2
        assert len(concat_video_results[0]['img_metas']) == 2
        assert concat_video_results[0]['gt_bboxes'].ndim == 2
        assert concat_video_results[0]['gt_bboxes'].shape[1] == 5
        assert concat_video_results[0]['gt_bboxes'].shape[0] == (
            num_ref_imgs * num_objects)

        concat_ref = dict(type='ConcatVideoReferences')
        concat_ref = build_from_cfg(concat_ref, PIPELINES)
        results = concat_ref(results)
        assert len(results) == 2
        assert results[0] == key_results
        assert results[1]['img'].ndim == 4
        assert results[1]['img'].shape[3] == 2
        assert len(results[1]['img_metas']) == 2
        assert results[1]['gt_bboxes'].ndim == 2
        assert results[1]['gt_bboxes'].shape[1] == 5
        assert results[1]['gt_bboxes'].shape[0] == (num_ref_imgs * num_objects)

        ref_prefix = 'ref'
        bundle = dict(type='SeqDefaultFormatBundle', ref_prefix=ref_prefix)
        bundle = build_from_cfg(bundle, PIPELINES)
        results = bundle(results)
        for key in results:
            if ref_prefix not in key:
                assert f'{ref_prefix}_{key}' in results
