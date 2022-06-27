# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase
from unittest.mock import MagicMock

import torch
from mmdet.core.bbox.demodata import random_boxes
from parameterized import parameterized

from mmtrack.registry import MODELS, TASK_UTILS
from mmtrack.utils import register_all_modules
from ..utils import _demo_mm_inputs, _get_model_cfg


class TestTracktorTracker(TestCase):

    @classmethod
    def setUpClass(cls):
        register_all_modules(init_default_scope=True)
        cls.num_objs = 30

    @parameterized.expand(
        ['mot/tracktor/tracktor_faster-rcnn_r50_fpn_4e_mot17-private-half.py'])
    def test_init(self, cfg_file):
        cfg = _get_model_cfg(cfg_file)
        tracker = MODELS.build(cfg['tracker'])

        bboxes = random_boxes(self.num_objs, 512)
        labels = torch.zeros(self.num_objs)
        scores = torch.ones(self.num_objs)
        ids = torch.arange(self.num_objs)
        tracker.update(
            ids=ids, bboxes=bboxes, scores=scores, labels=labels, frame_ids=0)

        assert tracker.ids == list(ids)
        assert tracker.memo_items == [
            'ids', 'bboxes', 'scores', 'labels', 'frame_ids'
        ]

    @parameterized.expand(
        ['mot/tracktor/tracktor_faster-rcnn_r50_fpn_4e_mot17-private-half.py'])
    def test_track(self, cfg_file):
        img = torch.rand((1, 3, 256, 256))
        x = [torch.rand(1, 256, 4, 4)]

        cfg = _get_model_cfg(cfg_file)
        tracker = MODELS.build(cfg['tracker'])

        model = MagicMock()
        model.detector = MODELS.build(cfg['detector'])
        model.reid = MODELS.build(cfg['reid'])
        model.cmc = TASK_UTILS.build(cfg['motion'])
        model.with_linear_motion = False

        packed_inputs = _demo_mm_inputs(
            batch_size=1, frame_id=0, num_ref_imgs=0)
        data_sample = packed_inputs[0]['data_sample']
        data_sample.pred_det_instances = data_sample.gt_instances.clone()
        # add fake scores
        scores = torch.ones(5)
        data_sample.pred_det_instances.scores = torch.FloatTensor(scores)
        for frame_id in range(3):
            pred_track_instances = tracker.track(
                model=model,
                img=img,
                feats=x,
                data_sample=packed_inputs[0]['data_sample'],
                data_preprocessor=cfg['data_preprocessor'])

            bboxes = pred_track_instances.bboxes
            labels = pred_track_instances.labels
            ids = pred_track_instances.instances_id

            assert bboxes.shape[1] == 4
            assert bboxes.shape[0] == labels.shape[0]
            assert bboxes.shape[0] == ids.shape[0]
