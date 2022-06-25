# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase
from unittest.mock import MagicMock

import torch
from mmdet.core.bbox.demodata import random_boxes
from parameterized import parameterized

from mmtrack.registry import MODELS, TASK_UTILS
from mmtrack.utils import register_all_modules
from ..utils import _demo_mm_inputs, _get_model_cfg


class TestSORTTracker(TestCase):

    @classmethod
    def setUpClass(cls):
        register_all_modules(init_default_scope=True)
        cls.num_objs = 30

    @parameterized.expand(
        ['mot/deepsort/deepsort_faster-rcnn_fpn_4e_mot17-private-half.py'])
    def test_init(self, cfg_file):
        cfg = _get_model_cfg(cfg_file)
        tracker = MODELS.build(cfg['tracker'])
        tracker.kf = TASK_UTILS.build(cfg['motion'])

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
        ['mot/deepsort/deepsort_faster-rcnn_fpn_4e_mot17-private-half.py'])
    def test_track(self, cfg_file):
        img = torch.rand((1, 3, 128, 128))

        cfg = _get_model_cfg(cfg_file)
        tracker = MODELS.build(cfg['tracker'])
        tracker.kf = TASK_UTILS.build(cfg['motion'])

        model = MagicMock()
        model.reid = MODELS.build(cfg['reid'])
        model.motion = TASK_UTILS.build(cfg['motion'])

        with torch.no_grad():
            for frame_id in range(3):
                packed_inputs = _demo_mm_inputs(
                    batch_size=1, frame_id=frame_id, num_ref_imgs=0)
                data_sample = packed_inputs[0]['data_sample']
                data_sample.pred_det_instances = \
                    data_sample.gt_instances.clone()
                # add fake scores
                scores = torch.ones(5)
                data_sample.pred_det_instances.scores = torch.FloatTensor(
                    scores)

                track_data_sample = tracker.track(
                    model=model,
                    img=img,
                    feats=None,
                    data_sample=packed_inputs[0]['data_sample'],
                    data_preprocessor=cfg['data_preprocessor'])
                pred_track_instances = track_data_sample.get(
                    'pred_track_instances', None)
                bboxes = pred_track_instances.bboxes
                labels = pred_track_instances.labels
                ids = pred_track_instances.instances_id

                assert bboxes.shape[1] == 4
                assert bboxes.shape[0] == labels.shape[0]
                assert bboxes.shape[0] == ids.shape[0]
