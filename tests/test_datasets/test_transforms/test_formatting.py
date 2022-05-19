# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np
import torch
from mmengine.data import LabelData

from mmtrack.core import ReIDDataSample
from mmtrack.datasets.transforms import (ConcatSameTypeFrames, PackReIDInputs,
                                         PackTrackInputs)


class TestConcatSameTypeFrames:

    def setup_class(cls):
        cls.img = np.zeros((100, 100, 3))
        cls.gt_bboxes = np.zeros((2, 4))
        cls.gt_bboxes_labels = np.zeros((2, ))
        cls.frame_id = 0
        cls.results = dict(
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
            frame_id=[cls.frame_id, cls.frame_id, cls.frame_id])

        cls.ref_prefix = 'ref'
        cls.concat_frames = ConcatSameTypeFrames(
            num_key_frames=1, ref_prefix=cls.ref_prefix)

    def test_transform(self):
        concat_results = self.concat_frames(self.results)
        assert isinstance(concat_results, dict)
        for key in self.results:
            assert key in concat_results
            assert f'{self.ref_prefix}_{key}' in concat_results

        assert list(concat_results['img'].shape) == list(self.img.shape)
        assert list(
            concat_results['ref_img'].shape) == list(self.img.shape) + [2]

        assert concat_results['gt_bboxes'].ndim == 2
        assert concat_results['gt_bboxes'].shape[1] == 4
        assert concat_results['ref_gt_bboxes'].ndim == 2
        assert concat_results['ref_gt_bboxes'].shape[1] == 5

        assert concat_results['gt_bboxes_labels'].ndim == 1
        assert concat_results['ref_gt_bboxes_labels'].ndim == 2
        assert concat_results['ref_gt_bboxes_labels'].shape[1] == 2

        assert concat_results['frame_id'] == self.frame_id
        assert concat_results['ref_frame_id'] == [self.frame_id, self.frame_id]


class TestPackTrackInputs:

    def setup_class(cls):
        cls.img = np.zeros((100, 100, 3))
        cls.gt_bboxes = np.zeros((2, 4))
        cls.gt_bboxes_labels = np.zeros((2, ))
        cls.frame_id = 0
        cls.results = dict(
            img=cls.img.copy(),
            gt_bboxes=cls.gt_bboxes.copy(),
            gt_bboxes_labels=cls.gt_bboxes_labels.copy(),
            frame_id=cls.frame_id,
            ref_img=cls.img.copy()[..., None],
            ref_gt_bboxes=np.concatenate((np.zeros(
                (2, 1)), cls.gt_bboxes.copy()),
                                         axis=1),
            ref_gt_bboxes_labels=np.concatenate((np.zeros(
                (2, 1)), cls.gt_bboxes_labels.copy()[..., None]),
                                                axis=1),
            ref_frame_id=[cls.frame_id, cls.frame_id])

        cls.ref_prefix = 'ref'
        cls.pack_track_inputs = PackTrackInputs(ref_prefix=cls.ref_prefix)

    def test_transform(self):
        track_results = self.pack_track_inputs(self.results)
        assert isinstance(track_results, dict)

        inputs = track_results['inputs']
        for key in ['img', 'ref_img']:
            assert isinstance(inputs[key], torch.Tensor)

        track_data_sample = track_results['data_sample']
        assert track_data_sample.metainfo['frame_id'] == self.frame_id
        assert track_data_sample.metainfo['ref_frame_id'] == [
            self.frame_id, self.frame_id
        ]

        assert track_data_sample.gt_instances.bboxes.shape == (2, 4)
        assert track_data_sample.ref_gt_instances.bboxes.shape == (2, 5)


class TestPackReIDInputs(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.results = dict(
            img=np.random.randn(256, 128, 3),
            gt_label=0,
            img_path='',
            ori_height=128,
            ori_width=128,
            img_shape=(256, 128),
            scale=(128, 256),
            scale_factor=(1., 2.),
            flip=False,
            flip_direction=None)
        cls.pack_reid_inputs = PackReIDInputs()

    def test_transform(self):
        results = self.pack_reid_inputs(self.results)
        self.assertIn('inputs', results)
        self.assertIsInstance(results['inputs']['img'], torch.Tensor)
        self.assertIn('data_sample', results)
        data_sample = results['data_sample']
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
