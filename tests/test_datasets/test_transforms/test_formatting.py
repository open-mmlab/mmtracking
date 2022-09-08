# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from unittest import TestCase

import numpy as np
import torch
from mmdet.structures.mask import BitmapMasks
from mmengine.structures import LabelData

from mmtrack.datasets.transforms import (CheckPadMaskValidity, PackReIDInputs,
                                         PackTrackInputs)
from mmtrack.structures import ReIDDataSample


class TestPackTrackInputs:

    def setup_class(cls):
        cls.H, cls.W = 100, 120
        cls.img = np.zeros((cls.H, cls.W, 3))
        cls.gt_bboxes = np.zeros((2, 4))
        cls.gt_bboxes_labels = np.zeros((2, ))
        cls.gt_masks = BitmapMasks(
            np.random.rand(2, cls.H, cls.W), height=cls.H, width=cls.W)
        cls.gt_instances_id = np.ones((2, ), dtype=np.int32)
        cls.padding_mask = np.zeros((cls.H, cls.W), dtype=np.int8)
        cls.frame_id = 0
        cls.scale_factor = 2.0
        cls.flip = False
        cls.ori_shape = (cls.H, cls.W)
        cls.results_1 = dict(
            img=[cls.img.copy(),
                 cls.img.copy(),
                 cls.img.copy()],
            gt_bboxes=[
                cls.gt_bboxes.copy(),
                cls.gt_bboxes.copy(),
                cls.gt_bboxes.copy()
            ],
            gt_bboxes_labels=[
                cls.gt_bboxes_labels.copy(),
                cls.gt_bboxes_labels.copy(),
                cls.gt_bboxes_labels.copy()
            ],
            gt_masks=[
                deepcopy(cls.gt_masks),
                deepcopy(cls.gt_masks),
                deepcopy(cls.gt_masks)
            ],
            gt_instances_id=[
                cls.gt_instances_id.copy(),
                cls.gt_instances_id.copy(),
                cls.gt_instances_id.copy(),
            ],
            frame_id=[cls.frame_id] * 3,
            ori_shape=[(cls.H, cls.W)] * 3,
            height=[cls.H] * 3,
            width=[cls.W] * 3,
            scale_factor=[cls.scale_factor] * 3,
            flip=[cls.flip] * 3,
            padding_mask=[
                cls.padding_mask.copy(),
                cls.padding_mask.copy(),
                cls.padding_mask.copy()
            ])

        cls.results_2 = deepcopy(cls.results_1)
        cls.results_2.update(
            dict(gt_ignore_flags=[np.array([0, 1], dtype=np.bool)] * 3))

        cls.results_3 = dict(
            img=cls.img.copy(),
            gt_bboxes=cls.gt_bboxes.copy(),
            gt_bboxes_labels=cls.gt_bboxes_labels.copy(),
            gt_masks=deepcopy(cls.gt_masks),
            gt_instances_id=cls.gt_instances_id.copy(),
            frame_id=cls.frame_id,
            ori_shape=(cls.H, cls.W),
            height=cls.H,
            width=cls.W,
            scale_factor=cls.scale_factor,
            flip=cls.flip,
            padding_mask=cls.padding_mask.copy())

        cls.ref_prefix = 'ref'
        cls.meta_keys = ('frame_id', 'ori_shape', 'scale_factor', 'flip')
        cls.pack_track_inputs = PackTrackInputs(
            num_key_frames=1,
            ref_prefix=cls.ref_prefix,
            meta_keys=cls.meta_keys,
            pack_single_img=False)

    def test_transform_without_ignore(self):
        self.pack_track_inputs.pack_single_img = False
        track_results = self.pack_track_inputs(self.results_1)
        assert isinstance(track_results, dict)

        inputs = track_results['inputs']
        assert isinstance(inputs['img'], torch.Tensor)
        assert inputs['img'].shape == (1, 3, self.H, self.W)
        assert isinstance(inputs['ref_img'], torch.Tensor)
        assert inputs['ref_img'].shape == (2, 3, self.H, self.W)

        track_data_sample = track_results['data_samples']

        assert track_data_sample.gt_instances.bboxes.shape == (2, 4)
        assert track_data_sample.ref_gt_instances.bboxes.shape == (4, 4)

        assert track_data_sample.gt_instances.labels.shape == (2, )
        assert track_data_sample.ref_gt_instances.labels.shape == (4, )

        assert track_data_sample.gt_instances.instances_id.shape == (2, )
        assert track_data_sample.ref_gt_instances.instances_id.shape == (4, )

        assert (track_data_sample.gt_instances.map_instances_to_img_idx ==
                torch.tensor([0, 0], dtype=torch.int32)).all()
        assert (track_data_sample.ref_gt_instances.map_instances_to_img_idx ==
                torch.tensor([0, 0, 1, 1], dtype=torch.int32)).all()

        assert len(track_data_sample.gt_instances.masks) == 2
        assert track_data_sample.gt_instances.masks.height == self.H
        assert track_data_sample.gt_instances.masks.width == self.W
        assert len(track_data_sample.ref_gt_instances.masks) == 4
        assert track_data_sample.ref_gt_instances.masks.height == self.H
        assert track_data_sample.ref_gt_instances.masks.width == self.W

        track_data_sample.padding_mask.shape == (1, self.H, self.W)
        track_data_sample.ref_padding_mask.shape == (2, self.H, self.W)

        for key in self.meta_keys:
            assert track_data_sample.metainfo[key] == getattr(self, key)
            assert track_data_sample.metainfo[f'ref_{key}'] == [
                getattr(self, key)
            ] * 2

    def test_transform_with_ignore(self):
        self.pack_track_inputs.pack_single_img = False
        track_results = self.pack_track_inputs(self.results_2)
        assert isinstance(track_results, dict)

        inputs = track_results['inputs']
        assert isinstance(inputs['img'], torch.Tensor)
        assert inputs['img'].shape == (1, 3, self.H, self.W)
        assert isinstance(inputs['ref_img'], torch.Tensor)
        assert inputs['ref_img'].shape == (2, 3, self.H, self.W)

        track_data_sample = track_results['data_samples']

        assert track_data_sample.gt_instances.bboxes.shape == (1, 4)
        assert track_data_sample.ref_gt_instances.bboxes.shape == (2, 4)

        assert track_data_sample.gt_instances.labels.shape == (1, )
        assert track_data_sample.ref_gt_instances.labels.shape == (2, )

        assert track_data_sample.gt_instances.instances_id.shape == (1, )
        assert track_data_sample.ref_gt_instances.instances_id.shape == (2, )

        assert (track_data_sample.gt_instances.map_instances_to_img_idx ==
                torch.tensor([0], dtype=torch.int32)).all()
        assert (track_data_sample.ref_gt_instances.map_instances_to_img_idx ==
                torch.tensor([0, 1], dtype=torch.int32)).all()

        assert len(track_data_sample.gt_instances.masks) == 1
        assert track_data_sample.gt_instances.masks.height == self.H
        assert track_data_sample.gt_instances.masks.width == self.W
        assert len(track_data_sample.ref_gt_instances.masks) == 2
        assert track_data_sample.ref_gt_instances.masks.height == self.H
        assert track_data_sample.ref_gt_instances.masks.width == self.W

        track_data_sample.padding_mask.shape == (1, self.H, self.W)
        track_data_sample.ref_padding_mask.shape == (2, self.H, self.W)

        for key in self.meta_keys:
            assert track_data_sample.metainfo[key] == getattr(self, key)
            assert track_data_sample.metainfo[f'ref_{key}'] == [
                getattr(self, key)
            ] * 2

    def test_transform_test_mode(self):
        self.pack_track_inputs.pack_single_img = True
        track_results = self.pack_track_inputs(self.results_3)
        assert isinstance(track_results, dict)

        inputs = track_results['inputs']
        assert isinstance(inputs['img'], torch.Tensor)
        assert inputs['img'].shape == (1, 3, self.H, self.W)

        track_data_sample = track_results['data_samples']

        assert track_data_sample.gt_instances.bboxes.shape == (2, 4)

        assert track_data_sample.gt_instances.labels.shape == (2, )

        assert track_data_sample.gt_instances.instances_id.shape == (2, )

        assert (track_data_sample.gt_instances.map_instances_to_img_idx ==
                torch.tensor([0], dtype=torch.int32)).all()

        assert len(track_data_sample.gt_instances.masks) == 2
        assert track_data_sample.gt_instances.masks.height == self.H
        assert track_data_sample.gt_instances.masks.width == self.W

        track_data_sample.padding_mask.shape == (1, self.H, self.W)

        for key in self.meta_keys:
            assert track_data_sample.metainfo[key] == getattr(self, key)


class TestPackReIDInputs(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.results = dict(
            img=np.random.randn(256, 128, 3),
            gt_label=0,
            img_path='',
            ori_shape=(128, 128),
            img_shape=(256, 128),
            scale=(128, 256),
            scale_factor=(1., 2.),
            flip=False,
            flip_direction=None)
        cls.pack_reid_inputs = PackReIDInputs(
            meta_keys=('flip', 'flip_direction'))

    def test_transform(self):
        results = self.pack_reid_inputs(self.results)
        self.assertIn('inputs', results)
        self.assertIsInstance(results['inputs'], torch.Tensor)
        self.assertIn('data_samples', results)
        data_sample = results['data_samples']
        self.assertIsInstance(data_sample, ReIDDataSample)
        self.assertIsInstance(data_sample.gt_label, LabelData)
        self.assertEqual(data_sample.img_path, '')
        self.assertEqual(data_sample.ori_shape, (128, 128))
        self.assertEqual(data_sample.img_shape, (256, 128))
        self.assertEqual(data_sample.scale, (128, 256))
        self.assertEqual(data_sample.scale_factor, (1., 2.))
        self.assertEqual(data_sample.flip, False)
        self.assertIsNone(data_sample.flip_direction)

    def test_repr(self):
        self.assertEqual(
            repr(self.pack_reid_inputs),
            f'PackReIDInputs(meta_keys={self.pack_reid_inputs.meta_keys})')


class TestCheckPadMaskValidity:

    def setup_class(cls):
        dummy = np.zeros((50, 50, 3))
        cls.results = dict(
            img=[dummy.copy(), dummy.copy(),
                 dummy.copy()],
            padding_mask=[dummy.copy(),
                          dummy.copy(),
                          dummy.copy()])

        cls.check_pad_mask_validity = CheckPadMaskValidity(stride=16)

    def test_transform(self):
        results = self.check_pad_mask_validity(self.results)
        assert results is not None
        self.results['padding_mask'][1] = np.ones((50, 50, 3))
        results = self.check_pad_mask_validity(self.results)
        assert results is None
