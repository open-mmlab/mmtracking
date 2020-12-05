import copy
import numpy as np
import os.path as osp
import pytest
from mmcv.utils import build_from_cfg
from mmdet.core.bbox.demodata import random_boxes

from mmtrack.datasets import PIPELINES


class TestTransforms(object):

    @classmethod
    def setup_class(cls):
        cls.data_prefix = osp.join(osp.dirname(__file__), '../assets')

        img_names = ['image_1.jpg', 'image_2.jpg']
        results = [
            dict(img_prefix=cls.data_prefix, img_info=dict(filename=name))
            for name in img_names
        ]
        load = build_from_cfg(
            dict(type='LoadMultiImagesFromFile', to_float32=True), PIPELINES)
        cls.results = load(results)

    def test_seq_resize(self):
        results = copy.deepcopy(self.results)
        transform = dict(
            type='SeqResize', img_scale=(512, 1024), keep_ratio=True)
        seq_resize = build_from_cfg(transform, PIPELINES)

        results = seq_resize(results)
        assert results[0]['img'].shape == (512, 1024, 3)
        assert results[1]['img'].shape == (512, 1024, 3)

    def test_seq_flip(self):

        transform = dict(
            type='SeqRandomFlip', share_params=True, flip_ratio=0.5)
        flip_module = build_from_cfg(transform, PIPELINES)

        for i in range(8):
            results = copy.deepcopy(self.results)
            results = flip_module(results)
            assert results[0]['flip'] == results[1]['flip']
            assert results[0]['flip_direction'] == results[1]['flip_direction']

        cases = [False, False]
        transform = dict(
            type='SeqRandomFlip', share_params=False, flip_ratio=0.5)
        flip_module = build_from_cfg(transform, PIPELINES)
        for i in range(8):
            results = copy.deepcopy(self.results)
            results = flip_module(results)
            if results[0]['flip'] == results[1]['flip']:
                cases[0] = True
            else:
                cases[1] = True
        assert cases[0] is True
        assert cases[1] is True

    def test_seq_pad(self):
        results = copy.deepcopy(self.results)

        transform = dict(type='SeqPad', size_divisor=32)
        transform = build_from_cfg(transform, PIPELINES)
        results = transform(results)

        for result in results:
            img_shape = result['img'].shape
            assert img_shape[0] % 32 == 0
            assert img_shape[1] % 32 == 0

        resize_transform = dict(
            type='SeqResize', img_scale=(1333, 800), keep_ratio=True)
        resize_module = build_from_cfg(resize_transform, PIPELINES)
        results = resize_module(results)
        results = transform(results)
        for result in results:
            img_shape = result['img'].shape
            assert img_shape[0] % 32 == 0
            assert img_shape[1] % 32 == 0

    def test_seq_normalize(self):
        results = copy.deepcopy(self.results)
        img_norm_cfg = dict(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True)
        transform = dict(type='SeqNormalize', **img_norm_cfg)
        transform = build_from_cfg(transform, PIPELINES)
        results = transform(results)

        mean = np.array(img_norm_cfg['mean'])
        std = np.array(img_norm_cfg['std'])
        for i, result in enumerate(results):
            converted_img = (self.results[i]['img'][..., ::-1] - mean) / std
            assert np.allclose(result['img'], converted_img)

    def test_seq_random_crop(self):
        # test assertion for invalid random crop
        with pytest.raises(AssertionError):
            transform = dict(
                type='SeqRandomCrop', crop_size=(-1, 0), share_params=False)
            build_from_cfg(transform, PIPELINES)

        crop_size = (256, 384)
        transform = dict(
            type='SeqRandomCrop', crop_size=crop_size, share_params=False)
        crop_module = build_from_cfg(transform, PIPELINES)

        results = copy.deepcopy(self.results)
        for res in results:
            res['gt_bboxes'] = random_boxes(8, 256)
            res['gt_labels'] = np.random.randint(8)
            res['gt_instance_ids'] = np.random.randint(8)
            res['gt_bboxes_ignore'] = random_boxes(2, 256)

        outs = crop_module(results)
        assert len(outs) == len(results)
        for res in results:
            assert res['img'].shape[:2] == crop_size
            # All bboxes should be reserved after crop
            assert res['img_shape'][:2] == crop_size
            assert res['gt_bboxes'].shape[0] == 8
            assert res['gt_bboxes_ignore'].shape[0] == 2
        assert outs[0]['img_info']['crop_offsets'] != outs[1]['img_info'][
            'crop_offsets']

        crop_module.share_params = True
        outs = crop_module(results)
        assert outs[0]['img_info']['crop_offsets'] == outs[1]['img_info'][
            'crop_offsets']

    def test_seq_color_jitter(self):
        results = self.results.copy()
        transform = dict(type='SeqPhotoMetricDistortion', share_params=False)
        transform = build_from_cfg(transform, PIPELINES)

        outs = transform(results)
        assert outs[0]['img_info']['color_jitter'] != outs[1]['img_info'][
            'color_jitter']

        transform.share_params = True
        outs = transform(results)
        assert outs[0]['img_info']['color_jitter'] == outs[1]['img_info'][
            'color_jitter']
