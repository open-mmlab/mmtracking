# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import shutil
import time
from unittest import TestCase
from unittest.mock import Mock

import torch
from mmengine.structures import InstanceData

from mmtrack.engine import TrackVisualizationHook
from mmtrack.structures import TrackDataSample
from mmtrack.visualization import TrackLocalVisualizer


class TestVisualizationHook(TestCase):

    def setUp(self) -> None:
        TrackLocalVisualizer.get_instance('visualizer')
        # pseudo data_batch
        self.data_batch = dict(data_samples=None, inputs=None)

        pred_instances_data = dict(
            bboxes=torch.tensor([[100, 100, 200, 200], [150, 150, 400, 200]]),
            instances_id=torch.tensor([1, 2]),
            labels=torch.tensor([0, 1]),
            scores=torch.tensor([0.955, 0.876]))
        pred_instances = InstanceData(**pred_instances_data)
        track_data_sample = TrackDataSample()
        track_data_sample.pred_track_instances = pred_instances
        track_data_sample.gt_instances = pred_instances
        track_data_sample.set_metainfo(
            dict(
                img_path=osp.join(
                    osp.dirname(__file__), '../../data/image_1.jpg')))
        self.outputs = [track_data_sample]

    def test_after_val_iter(self):
        runner = Mock()
        runner.iter = 1
        hook = TrackVisualizationHook(interval=10, draw=True)
        hook.after_val_iter(runner, 9, self.data_batch, self.outputs)

    def test_after_test_iter(self):
        runner = Mock()
        runner.iter = 1
        hook = TrackVisualizationHook(interval=10, draw=True)
        hook.after_val_iter(runner, 9, self.data_batch, self.outputs)

        # test test_out_dir
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        test_out_dir = timestamp + '1'
        runner.work_dir = timestamp
        runner.timestamp = '1'
        hook = TrackVisualizationHook(
            interval=10, draw=True, test_out_dir=test_out_dir)
        hook.after_test_iter(runner, 9, self.data_batch, self.outputs)
        self.assertTrue(os.path.exists(f'{timestamp}/1/{test_out_dir}'))
        shutil.rmtree(f'{timestamp}')
