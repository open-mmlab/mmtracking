# Copyright (c) OpenMMLab. All rights reserved.
import os
from unittest import TestCase

import torch
from mmengine.structures import InstanceData

from mmtrack.evaluation import MOTChallengeMetrics


class TestMOTChallengeMetrics(TestCase):

    def test_init(self):
        with self.assertRaisesRegex(KeyError, 'metric unknown is not'):
            MOTChallengeMetrics(metric='unknown')
        with self.assertRaises(AssertionError):
            MOTChallengeMetrics(benchmark='MOT21')

    @staticmethod
    def _get_predictions_demo():
        instances = [{
            'bbox_label': 0,
            'bbox': [0, 0, 100, 100],
            'ignore_flag': 0,
            'instance_id': 1,
            'mot_conf': 1.0,
            'category_id': 1,
            'visibility': 1.0
        }, {
            'bbox_label': 0,
            'bbox': [0, 0, 100, 100],
            'ignore_flag': 0,
            'instance_id': 2,
            'mot_conf': 1.0,
            'category_id': 1,
            'visibility': 1.0
        }]
        sep = os.sep
        pred_instances_data = dict(
            bboxes=torch.tensor([
                [0, 0, 100, 100],
                [0, 0, 100, 40],
            ]),
            instances_id=torch.tensor([1, 2]),
            scores=torch.tensor([1.0, 1.0]))
        pred_instances = InstanceData(**pred_instances_data)
        predictions = [
            dict(
                pred_track_instances=pred_instances,
                frame_id=0,
                video_length=1,
                img_id=1,
                img_path=f'xxx{sep}MOT17-09-DPM{sep}img1{sep}000001.jpg',
                instances=instances)
        ]
        return predictions

    def _test_evaluate(self, format_only):
        """Test using the metric in the same way as Evaluator."""
        metric = MOTChallengeMetrics(
            metric=['HOTA', 'CLEAR', 'Identity'], format_only=format_only)
        metric.dataset_meta = {'classes': ('pedestrian', )}
        data_batch = dict(input=None, data_samples=None)
        predictions = self._get_predictions_demo()
        metric.process(data_batch, predictions)
        eval_results = metric.evaluate()
        return eval_results

    def test_evaluate(self):
        eval_results = self._test_evaluate(False)
        target = {
            'motchallenge-metric/IDF1': 0.5,
            'motchallenge-metric/MOTA': 0,
            'motchallenge-metric/HOTA': 0.755,
            'motchallenge-metric/IDSW': 0,
        }
        for key in target:
            assert eval_results[key] - target[key] < 1e-3

    def test_evaluate_format_only(self):
        eval_results = self._test_evaluate(True)
        assert eval_results == dict()
