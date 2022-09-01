# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase
from unittest.mock import MagicMock

import torch

from mmtrack.registry import MODELS, TASK_UTILS
from mmtrack.testing import demo_mm_inputs, random_boxes
from mmtrack.utils import register_all_modules


class TestByteTracker(TestCase):

    @classmethod
    def setUpClass(cls):
        register_all_modules(init_default_scope=True)
        cfg = dict(
            type='ByteTracker',
            obj_score_thrs=dict(high=0.6, low=0.1),
            init_track_thr=0.7,
            weight_iou_with_det_scores=True,
            match_iou_thrs=dict(high=0.1, low=0.5, tentative=0.3),
            num_tentatives=3,
            num_frames_retain=30)
        cls.tracker = MODELS.build(cfg)
        cls.tracker.kf = TASK_UTILS.build(dict(type='KalmanFilter'))
        cls.num_frames_retain = cfg['num_frames_retain']
        cls.num_objs = 30

    def test_init(self):
        bboxes = random_boxes(self.num_objs, 512)
        labels = torch.zeros(self.num_objs)
        scores = torch.ones(self.num_objs)
        ids = torch.arange(self.num_objs)
        self.tracker.update(
            ids=ids, bboxes=bboxes, scores=scores, labels=labels, frame_ids=0)

        assert self.tracker.ids == list(ids)
        assert self.tracker.memo_items == [
            'ids', 'bboxes', 'scores', 'labels', 'frame_ids'
        ]

    def test_track(self):
        img_size = 64
        img = torch.rand((1, 3, img_size, img_size))

        model = MagicMock()

        for frame_id in range(3):
            packed_inputs = demo_mm_inputs(
                batch_size=1, frame_id=frame_id, num_ref_imgs=0)
            data_sample = packed_inputs['data_samples'][0]
            data_sample.pred_det_instances = data_sample.gt_instances.clone()
            # add fake scores
            scores = torch.ones(5)
            data_sample.pred_det_instances.scores = torch.FloatTensor(scores)

            pred_track_instances = self.tracker.track(
                model=model,
                img=img,
                feats=None,
                data_sample=packed_inputs['data_samples'][0])

            bboxes = pred_track_instances.bboxes
            labels = pred_track_instances.labels
            ids = pred_track_instances.instances_id

            assert bboxes.shape[1] == 4
            assert bboxes.shape[0] == labels.shape[0]
            assert bboxes.shape[0] == ids.shape[0]
