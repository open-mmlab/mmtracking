# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import tempfile
from unittest import TestCase

import numpy as np
import torch
from mmcv import list_from_file

from mmtrack.registry import METRICS
from mmtrack.utils import register_all_modules

SOT_DATA_PREFIX = osp.join(osp.dirname(__file__), '../data/demo_sot_data')


class TestSOTMetric(TestCase):

    @classmethod
    def setUpClass(cls):
        register_all_modules(init_default_scope=True)
        cls.tmp_dir = tempfile.TemporaryDirectory()
        cls.outfile_prefix = f'{cls.tmp_dir.name}/test'
        cls.sot_metric = METRICS.build(
            dict(
                type='SOTMetric',
                outfile_prefix=cls.outfile_prefix,
            ))

    @classmethod
    def tearDownClass(cls):
        cls.tmp_dir.cleanup()

    def _create_eval_inputs(self, pred_file, gt_file):
        data_root = osp.join(SOT_DATA_PREFIX, 'trackingnet', 'TRAIN_0')
        gts, preds = [], []
        for video_id, video_name in enumerate(['video-1', 'video-2']):
            pred_bboxes = list_from_file(
                osp.join(data_root, video_name, pred_file))
            gt_bboxes = np.loadtxt(
                osp.join(data_root, video_name, gt_file), delimiter=',')

            for i, (pred_bbox,
                    gt_bbox) in enumerate(zip(pred_bboxes, gt_bboxes)):
                pred_bbox = list(map(lambda x: float(x), pred_bbox.split(',')))
                pred_track_instances = dict(
                    bboxes=torch.Tensor(pred_bbox)[None])
                if len(gt_bbox) == 4:
                    gt_bbox[2:] += gt_bbox[:2]
                data_sample = dict(
                    video_id=video_id,
                    frame_id=i,
                    img_path=osp.join(data_root, video_name, 'demo.jpg'),
                    ori_shape=(256, 512),
                    video_length=25,
                    instances=[dict(bbox=gt_bbox, visible=True)])
                preds.append(dict(pred_track_instances=pred_track_instances))
                gts.append(dict(data_sample=data_sample))
        return (gts, preds)

    def test_evaluate(self):
        """Test using the metric in the same way as Evaluator."""
        metric_prefix = self.sot_metric.prefix

        # 1. OPE evaluation
        self.sot_metric.metrics = ['OPE']
        gts, preds = self._create_eval_inputs('track_results.txt',
                                              'gt_for_eval.txt')
        for gt, pred in zip(gts, preds):
            self.sot_metric.process([gt], [pred])
        eval_results = self.sot_metric.evaluate(size=50)
        assert round(eval_results[f'{metric_prefix}/success'], 4) == 67.5238
        assert eval_results[f'{metric_prefix}/norm_precision'] == 70.0
        assert eval_results[f'{metric_prefix}/precision'] == 50.0

        # 2. Format-only
        self.sot_metric.format_only = True
        for gt, pred in zip(gts, preds):
            self.sot_metric.process([gt], [pred])
        eval_results = self.sot_metric.evaluate(size=50)
        assert len(eval_results) == 0
        assert os.path.exists(f'{self.outfile_prefix}.zip')

        # 3. VOT evaluation
        self.sot_metric.format_only = False
        self.sot_metric.metrics = ['VOT']
        self.sot_metric.metric_options['interval'] = [1, 3]
        gts, preds = self._create_eval_inputs('vot2018_track_results.txt',
                                              'vot2018_gt_for_eval.txt')
        for gt, pred in zip(gts, preds):
            self.sot_metric.process([gt], [pred])
        eval_results = self.sot_metric.evaluate(size=50)
        assert abs(eval_results[f'{metric_prefix}/eao'] - 0.6661) < 0.0001
        assert round(eval_results[f'{metric_prefix}/accuracy'], 4) == 0.5826
        assert round(eval_results[f'{metric_prefix}/robustness'], 4) == 6.0
