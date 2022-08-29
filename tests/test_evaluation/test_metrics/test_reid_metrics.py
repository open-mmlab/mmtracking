# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmtrack.registry import METRICS
from mmtrack.structures import ReIDDataSample
from mmtrack.utils import register_all_modules


class TestReIDMetrics(TestCase):

    @classmethod
    def setUpClass(cls):
        register_all_modules(init_default_scope=True)

    def test_evaluate(self):
        """Test using the metric in the same way as Evaluator."""
        data_batch = [{
            'data_samples':
            ReIDDataSample().set_gt_label(i).to_dict()
        } for i in [0, 0, 1, 1, 1, 1]]
        pred_batch = [
            dict(pred_feature=torch.tensor(
                [1., .0, .1])),  # [x,√,x,x,x],R1=0,R5=1,AP=0.50
            dict(pred_feature=torch.tensor(
                [.8, .0, .0])),  # [x,√,x,x,x],R1=0,R5=1,AP=0.50
            dict(pred_feature=torch.tensor(
                [.1, 1., .1])),  # [√,√,x,√,x],R1=1,R5=1,AP≈0.92
            dict(pred_feature=torch.tensor(
                [.0, .9, .1])),  # [√,√,√,x,x],R1=1,R5=1,AP=1.00
            dict(pred_feature=torch.tensor(
                [.9, .1, .0])),  # [x,x,√,√,√],R1=0,R5=1,AP≈0.48
            dict(pred_feature=torch.tensor(
                [.0, .1, 1.])),  # [√,√,x,√,x],R1=1,R5=1,AP≈0.92
        ]
        metric = METRICS.build(
            dict(
                type='ReIDMetrics',
                metric=['mAP', 'CMC'],
                metric_options=dict(rank_list=[1, 5], max_rank=5),
            ))

        prefix = 'reid-metric'
        metric.process(data_batch, pred_batch)
        results = metric.evaluate(6)
        self.assertIsInstance(results, dict)
        self.assertEqual(results[f'{prefix}/mAP'], 0.719)
        self.assertEqual(results[f'{prefix}/R1'], 0.5)
        self.assertEqual(results[f'{prefix}/R5'], 1.0)
