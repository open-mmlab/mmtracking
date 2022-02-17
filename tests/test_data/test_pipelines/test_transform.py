# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp

import numpy as np
import pytest
from mmcv.utils import build_from_cfg
from mmdet.core.bbox.demodata import random_boxes

from mmtrack.datasets import PIPELINES


class TestTransforms(object):

    @classmethod
    def setup_class(cls):
        cls.data_prefix = osp.join(osp.dirname(__file__), '../../data')

        img_names = ['image_1.jpg', 'image_2.jpg']
        results = [
            dict(img_prefix=cls.data_prefix, img_info=dict(filename=name))
            for name in img_names
        ]
        load = build_from_cfg(
            dict(type='LoadMultiImagesFromFile', to_float32=True), PIPELINES)
        cls.results = load(results)

    def test_seq_crop_like_siamfc(self):
        results = copy.deepcopy(self.results)
        for res in results:
            res['gt_bboxes'] = random_boxes(1, 256)
            res['bbox_fields'] = ['gt_bboxes']

        transform = dict(
            type='SeqCropLikeSiamFC',
            context_amount=0.5,
            exemplar_size=127,
            crop_size=511)
        seq_crop_like_siamfc = build_from_cfg(transform, PIPELINES)

        results = seq_crop_like_siamfc(results)
        assert results[0]['img'].shape == (511, 511, 3)
        assert results[1]['img'].shape == (511, 511, 3)

    def test_seq_bbox_jitter(self):
        results = copy.deepcopy(self.results)
        for res in results:
            res['gt_bboxes'] = random_boxes(1, 256)
            res['bbox_fields'] = ['gt_bboxes']

        transform = dict(
            type='SeqBboxJitter',
            center_jitter_factor=[0, 4.5],
            scale_jitter_factor=[0, 0.5],
            crop_size_factor=[2, 5])
        seq_bbox_jitter = build_from_cfg(transform, PIPELINES)
        results = seq_bbox_jitter(results)
        assert results[0]['jittered_bboxes'].shape == (1, 4)
        assert results[1]['jittered_bboxes'].shape == (1, 4)

    def test_seq_crop_like_stark(self):
        results = copy.deepcopy(self.results)
        for res in results:
            res['gt_bboxes'] = random_boxes(1, 256)
            res['jittered_bboxes'] = np.array([[
                res['gt_bboxes'][0][0] - 1, res['gt_bboxes'][0][1] + 2,
                res['gt_bboxes'][0][2] + 2, res['gt_bboxes'][0][3] + 3
            ]])
            res['bbox_fields'] = ['gt_bboxes']

        transform = dict(
            type='SeqCropLikeStark',
            crop_size_factor=[2, 5],
            output_size=[128, 320])
        seq_crop_like_stark = build_from_cfg(transform, PIPELINES)
        results = seq_crop_like_stark(results)
        assert results[0]['img'].shape == (128, 128, 3)
        assert results[1]['img'].shape == (320, 320, 3)

    def test_seq_brightness_aug(self):
        results = copy.deepcopy(self.results)
        imgs_shape = [result['img'].shape for result in results]

        transform = dict(type='SeqBrightnessAug', jitter_range=0.2)
        seq_brightness_aug = build_from_cfg(transform, PIPELINES)

        results = seq_brightness_aug(results)
        assert results[0]['img'].shape == imgs_shape[0]
        assert results[1]['img'].shape == imgs_shape[1]

    def test_seq_gray_aug(self):
        results = copy.deepcopy(self.results)
        imgs_shape = [result['img'].shape for result in results]

        transform = dict(type='SeqGrayAug', prob=0.2)
        seq_gray_aug = build_from_cfg(transform, PIPELINES)

        results = seq_gray_aug(results)
        assert results[0]['img'].shape == imgs_shape[0]
        assert results[1]['img'].shape == imgs_shape[1]

    def test_seq_shift_scale_aug(self):
        results = copy.deepcopy(self.results)
        for res in results:
            res['gt_bboxes'] = random_boxes(1, 256).numpy()
            res['bbox_fields'] = ['gt_bboxes']

        transform = dict(
            type='SeqShiftScaleAug',
            target_size=[127, 255],
            shift=[4, 64],
            scale=[0.05, 0.18])
        seq_shift_scale_aug = build_from_cfg(transform, PIPELINES)

        results = seq_shift_scale_aug(results)
        assert results[0]['img'].shape == (127, 127, 3)
        assert results[1]['img'].shape == (255, 255, 3)

    def test_seq_color_aug(self):
        results = copy.deepcopy(self.results)
        imgs_shape = [result['img'].shape for result in results]

        transform = dict(
            type='SeqColorAug',
            prob=[1.0, 1.0],
            rgb_var=[[-0.55919361, 0.98062831, -0.41940627],
                     [1.72091413, 0.19879334, -1.82968581],
                     [4.64467907, 4.73710203, 4.88324118]])
        seq_color_aug = build_from_cfg(transform, PIPELINES)

        results = seq_color_aug(results)
        assert results[0]['img'].shape == imgs_shape[0]
        assert results[1]['img'].shape == imgs_shape[1]

    def test_seq_blur_aug(self):
        results = copy.deepcopy(self.results)
        imgs_shape = [result['img'].shape for result in results]

        transform = dict(type='SeqBlurAug', prob=[0.0, 0.2])
        seq_blur_aug = build_from_cfg(transform, PIPELINES)

        results = seq_blur_aug(results)
        assert results[0]['img'].shape == imgs_shape[0]
        assert results[1]['img'].shape == imgs_shape[1]

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
        for i in range(20):
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
