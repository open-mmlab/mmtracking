# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import MagicMock

import torch
from mmdet.core.bbox.demodata import random_boxes

from mmtrack.models import TRACKERS, KalmanFilter


class TestOCSORTTracker(object):

    @classmethod
    def setup_class(cls):
        cfg = dict(
            obj_score_thr=0.3,
            init_track_thr=0.7,
            weight_iou_with_det_scores=True,
            match_iou_thr=0.3,
            num_tentatives=3,
            vel_consist_weight=0.2,
            vel_delta_t=3,
            num_frames_retain=30)
        tracker = TRACKERS.get('OCSORTTracker')
        cls.tracker = tracker(**cfg)
        cls.tracker.kf = KalmanFilter()
        cls.num_objs = 5

    def test_track(self):
        img_size = 64
        img = torch.rand((1, 3, img_size, img_size))

        img_metas = [dict(scale_factor=1.0)]

        model = MagicMock()

        bboxes = random_boxes(self.num_objs, img_size)
        scores = torch.rand((self.num_objs, 1))
        bboxes = torch.cat((bboxes, scores), dim=1)

        labels = torch.arange(self.num_objs)

        for frame_id in range(3):
            bboxes, labels, ids = self.tracker.track(
                img, img_metas, model, bboxes, labels, frame_id, rescale=True)
            assert bboxes.shape[0] == labels.shape[0]
            assert bboxes.shape[0] == ids.shape[0]
