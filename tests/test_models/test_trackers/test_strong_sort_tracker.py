# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase
from unittest.mock import MagicMock

import torch
from parameterized import parameterized

from mmtrack.registry import MODELS, TASK_UTILS
from mmtrack.testing import demo_mm_inputs, get_model_cfg, random_boxes
from mmtrack.utils import register_all_modules


class TestStrongSORTTracker(TestCase):

    @classmethod
    def setUpClass(cls):
        register_all_modules(init_default_scope=True)
        cls.num_objs = 30

    @parameterized.expand([
        'mot/strongsort/strongsort_yolox_x_8xb4-80e_crowdhuman-mot17halftrain'
        '_test-mot17halfval.py'
    ])
    def test_init(self, cfg_file):
        cfg = get_model_cfg(cfg_file)
        tracker = MODELS.build(cfg['tracker'])
        tracker.kf = TASK_UTILS.build(cfg['kalman'])
        tracker.cmc = TASK_UTILS.build(cfg['cmc'])

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

    @parameterized.expand([
        'mot/strongsort/strongsort_yolox_x_8xb4-80e_crowdhuman-mot17halftrain'
        '_test-mot17halfval.py'
    ])
    def test_track(self, cfg_file):
        img = torch.rand((1, 3, 128, 128))

        cfg = get_model_cfg(cfg_file)
        tracker = MODELS.build(cfg['tracker'])
        tracker.kf = TASK_UTILS.build(cfg['kalman'])
        tracker.cmc = TASK_UTILS.build(cfg['cmc'])

        model = MagicMock()
        model.reid = MODELS.build(cfg['reid'])
        model.kalman = TASK_UTILS.build(cfg['kalman'])
        model.cmc = TASK_UTILS.build(cfg['cmc'])

        with torch.no_grad():
            for frame_id in range(3):
                packed_inputs = demo_mm_inputs(
                    batch_size=1, frame_id=frame_id, num_ref_imgs=0)
                data_sample = packed_inputs[0]['data_sample']
                data_sample.pred_det_instances = \
                    data_sample.gt_instances.clone()
                # add fake scores
                scores = torch.ones(5)
                data_sample.pred_det_instances.scores = \
                    torch.FloatTensor(scores)

                pred_track_instances = tracker.track(
                    model=model,
                    img=img,
                    feats=None,
                    data_sample=packed_inputs[0]['data_sample'],
                    data_preprocessor=cfg['data_preprocessor'])

                bboxes = pred_track_instances.bboxes
                labels = pred_track_instances.labels
                ids = pred_track_instances.instances_id

                assert bboxes.shape[1] == 4
                assert bboxes.shape[0] == labels.shape[0]
                assert bboxes.shape[0] == ids.shape[0]
