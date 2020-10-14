import copy
import os.path as osp

import numpy as np
from mmcv.utils import build_from_cfg

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
        load = build_from_cfg(dict(type='LoadMultiImagesFromFile'), PIPELINES)
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

    # def test_seq_random_crop(self):
    #     transform = dict(type='SeqRandomCrop', **img_norm_cfg)
    #     transform = build_from_cfg(transform, PIPELINES)
