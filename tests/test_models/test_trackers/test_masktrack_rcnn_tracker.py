# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import MagicMock

import torch
from mmdet.core.bbox.demodata import random_boxes

from mmtrack.models import TRACKERS


class TestBaseTracker(object):

    @classmethod
    def setup_class(cls):
        cfg = dict(
            det_score_coefficient=1.0,
            iou_coefficient=1.0,
            label_coefficient=1000.0)
        tracker = TRACKERS.get('MaskTrackRCNNTracker')
        cls.tracker = tracker(**cfg)
        cls.num_objs = 5

    def test_track(self):
        img = torch.rand((1, 3, 64, 64))

        img_metas = [dict(scale_factor=1.0)]

        model = MagicMock()
        model.track_head.extract_roi_feats = MagicMock(
            return_value=(torch.rand(5, 8, 7, 7), [5]))
        model.track_head.simple_test = MagicMock(
            return_value=torch.rand((self.num_objs, self.num_objs + 1)))

        feats = torch.rand((1, 8, 32, 32))

        bboxes = random_boxes(self.num_objs, 512)
        scores = torch.rand((self.num_objs, 1))
        bboxes = torch.cat((bboxes, scores), dim=1)

        labels = torch.arange(self.num_objs)

        masks = torch.zeros((self.num_objs, 100, 100))

        for frame_id in range(3):
            bboxes, labels, masks, ids = self.tracker.track(
                img,
                img_metas,
                model,
                feats,
                bboxes,
                labels,
                masks,
                frame_id,
                rescale=True)
            assert bboxes.shape[0] == self.num_objs
            assert labels.shape[0] == self.num_objs
            assert masks.shape == (self.num_objs, 100, 100)
            assert ids.shape[0] == self.num_objs
