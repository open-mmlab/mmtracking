# Copyright (c) OpenMMLab. All rights reserved.

import os.path as osp
import tempfile
from unittest import TestCase

import numpy as np
import torch

from mmtrack.registry import METRICS
from mmtrack.utils import register_all_modules


class TestTAOMetric(TestCase):

    @classmethod
    def setUpClass(cls):
        register_all_modules(init_default_scope=True)

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _create_dummy_results(self, track_id):
        bboxes = np.array([[100, 100, 150, 150]])
        scores = np.array([1.0])
        labels = np.array([0])
        instance_id = np.array([track_id])
        return dict(
            bboxes=torch.from_numpy(bboxes),
            scores=torch.from_numpy(scores),
            labels=torch.from_numpy(labels),
            instances_id=torch.from_numpy(instance_id))

    def test_format_only(self):
        outfile_prefix = f'{self.tmp_dir.name}/result'
        tao_metric = METRICS.build(
            dict(
                type='TAOMetric',
                format_only=True,
                outfile_prefix=outfile_prefix,
            ))
        dummy_pred = self._create_dummy_results(track_id=0)
        instances = [{
            'bbox_label': 0,
            'bbox': [100, 100, 150, 150],
            'ignore_flag': 0,
            'instance_id': 1,
        }]
        tao_metric.dataset_meta = dict(
            classes=['car', 'train'],
            categories={
                0: dict(id=0, name='car'),
                1: dict(id=1, name='train')
            })
        data_batch = dict(inputs=None, data_samples=None)
        data_samples = [
            dict(
                pred_track_instances=dummy_pred,
                pred_det_instances=dummy_pred,
                img_id=0,
                ori_shape=(720, 1280),
                frame_id=0,
                frame_index=0,
                video_id=1,
                video_length=1,
                instances=instances,
                neg_category_ids=[3, 4],
                not_exhaustive_category_ids=[1, 2])
        ]
        tao_metric.process(data_batch, data_samples)
        tao_metric.evaluate(size=1)
        assert osp.exists(f'{outfile_prefix}_track.json')
        assert osp.exists(f'{outfile_prefix}_det.json')

    def test_evaluate(self):
        """Test using the metric in the same way as Evaluator."""
        dummy_pred_1 = self._create_dummy_results(track_id=1)
        dummy_pred_2 = self._create_dummy_results(track_id=1)

        instances = [{
            'bbox_label': 0,
            'bbox': [100, 100, 150, 150],
            'ignore_flag': 0,
            'instance_id': 1
        }]
        tao_metric = METRICS.build(
            dict(
                type='TAOMetric',
                outfile_prefix=f'{self.tmp_dir.name}/test',
            ))

        tao_metric.dataset_meta = dict(
            classes=['car', 'train'],
            categories={
                0: dict(id=0, name='car'),
                1: dict(id=1, name='train')
            })
        data_batch = dict(inputs=None, data_samples=None)
        data_samples = [
            dict(
                pred_track_instances=dummy_pred_1,
                pred_det_instances=dummy_pred_1,
                img_id=0,
                ori_shape=(720, 1280),
                frame_id=0,
                frame_index=0,
                video_id=1,
                video_length=1,
                instances=instances,
                neg_category_ids=[3, 4],
                not_exhaustive_category_ids=[1, 2])
        ]
        tao_metric.process(data_batch, data_samples)

        data_samples = [
            dict(
                pred_track_instances=dummy_pred_2,
                pred_det_instances=dummy_pred_2,
                img_id=0,
                ori_shape=(720, 1280),
                frame_id=0,
                frame_index=0,
                video_id=1,
                video_length=1,
                instances=instances,
                neg_category_ids=[3, 4],
                not_exhaustive_category_ids=[1, 2])
        ]
        tao_metric.process(data_batch, data_samples)

        eval_results = tao_metric.evaluate(size=2)
        target = {
            'tao/track_AP': 1.0,
            'tao/track_AP50': 1.0,
            'tao/track_AP75': 1.0,
        }
        self.assertDictEqual(eval_results, target)
