# Copyright (c) OpenMMLab. All rights reserved.
import os
import tempfile
from unittest import TestCase

import numpy as np
import pycocotools.mask as mask_util
import torch

from mmtrack.registry import METRICS
from mmtrack.utils import register_all_modules


class TestYouTubeVISMetric(TestCase):

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
        dummy_mask = np.zeros((1, 720, 1280), dtype=np.uint8)
        dummy_mask[:, 100:150, 100:150] = 1
        return dict(
            bboxes=torch.from_numpy(bboxes),
            scores=torch.from_numpy(scores),
            labels=torch.from_numpy(labels),
            instance_id=torch.from_numpy(instance_id),
            masks=torch.from_numpy(dummy_mask))

    def test_format_only(self):
        outfile_prefix = f'{self.tmp_dir.name}/result'
        vis_metric = METRICS.build(
            dict(
                type='YouTubeVISMetric',
                format_only=True,
                outfile_prefix=outfile_prefix,
            ))
        dummy_pred = self._create_dummy_results(track_id=0)
        dummy_mask = np.zeros((720, 1280), order='F', dtype=np.uint8)
        dummy_mask[100:150, 100:150] = 1
        rle_mask = mask_util.encode(dummy_mask)
        rle_mask['counts'] = rle_mask['counts'].decode('utf-8')
        instances = [{
            'bbox_label': 0,
            'bbox': [100, 100, 150, 150],
            'ignore_flag': 0,
            'instance_id': 1,
            'mask': rle_mask,
        }]
        vis_metric.dataset_meta = dict(CLASSES=['car', 'train'])
        vis_metric.process([
            dict(
                inputs=None,
                data_sample={
                    'img_id': 1,
                    'ori_shape': (720, 1280),
                    'frame_id': 0,
                    'video_id': 1,
                    'video_length': 1,
                    'instances': instances
                })
        ], [dict(pred_track_instances=dummy_pred)])
        vis_metric.evaluate(size=1)
        assert os.path.exists(f'{outfile_prefix}.json')
        assert os.path.exists(f'{outfile_prefix}.submission_file.zip')

    def test_evaluate(self):
        """Test using the metric in the same way as Evaluator."""
        dummy_pred_1 = self._create_dummy_results(track_id=1)
        dummy_pred_2 = self._create_dummy_results(track_id=1)
        dummy_pred_3 = self._create_dummy_results(track_id=2)

        dummy_mask = np.zeros((720, 1280), order='F', dtype=np.uint8)
        dummy_mask[100:150, 100:150] = 1
        rle_mask = mask_util.encode(dummy_mask)
        rle_mask['counts'] = rle_mask['counts'].decode('utf-8')
        instances_1 = [{
            'bbox_label': 0,
            'bbox': [100, 100, 150, 150],
            'ignore_flag': 0,
            'instance_id': 1,
            'mask': rle_mask,
        }]
        instances_2 = [{
            'bbox_label': 0,
            'bbox': [100, 100, 150, 150],
            'ignore_flag': 0,
            'instance_id': 2,
            'mask': rle_mask,
        }]
        vis_metric = METRICS.build(
            dict(
                type='YouTubeVISMetric',
                outfile_prefix=f'{self.tmp_dir.name}/test',
            ))

        vis_metric.dataset_meta = dict(CLASSES=['car', 'train'])
        vis_metric.process([
            dict(
                inputs=None,
                data_sample={
                    'img_id': 1,
                    'ori_shape': (720, 1280),
                    'frame_id': 0,
                    'video_id': 1,
                    'video_length': 2,
                    'instances': instances_1
                })
        ], [dict(pred_track_instances=dummy_pred_1)])
        vis_metric.process([
            dict(
                inputs=None,
                data_sample={
                    'img_id': 2,
                    'ori_shape': (720, 1280),
                    'frame_id': 1,
                    'video_id': 1,
                    'video_length': 2,
                    'instances': instances_1
                })
        ], [dict(pred_track_instances=dummy_pred_2)])
        vis_metric.process([
            dict(
                inputs=None,
                data_sample={
                    'img_id': 3,
                    'ori_shape': (720, 1280),
                    'frame_id': 0,
                    'video_id': 2,
                    'video_length': 1,
                    'instances': instances_2
                })
        ], [dict(pred_track_instances=dummy_pred_3)])

        eval_results = vis_metric.evaluate(size=3)
        assert eval_results['youtube_vis/segm_mAP_copypaste'] \
               == '1.000 1.000 1.000 1.000 -1.000 -1.000'
