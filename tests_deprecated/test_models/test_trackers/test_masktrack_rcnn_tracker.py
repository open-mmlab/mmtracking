# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import MagicMock

import torch
from mmdet.core.bbox.demodata import random_boxes

from mmtrack.models import TRACKERS


class TestMaskTrackRCNNTracker(object):

    @classmethod
    def setup_class(cls):
        cfg = dict(
            match_weights=dict(det_score=1.0, iou=1.0, det_label=1000.0), )
        tracker = TRACKERS.get('MaskTrackRCNNTracker')
        cls.tracker = tracker(**cfg)
        cls.num_objs = 5

    def test_track(self):
        img_size, feats_channel = 64, 8
        img = torch.rand((1, 3, img_size, img_size))

        img_metas = [dict(scale_factor=1.0)]

        model = MagicMock()
        model.track_head.extract_roi_feats = MagicMock(
            return_value=(torch.rand(self.num_objs, feats_channel, 7, 7),
                          [self.num_objs]))
        model.track_head.simple_test = MagicMock(
            return_value=torch.rand((self.num_objs, self.num_objs + 1)))

        feats = torch.rand((1, feats_channel, img_size, img_size))

        bboxes = random_boxes(self.num_objs, img_size)
        scores = torch.rand((self.num_objs, 1))
        bboxes = torch.cat((bboxes, scores), dim=1)

        labels = torch.arange(self.num_objs)

        masks = torch.zeros((self.num_objs, img_size, img_size))

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
            assert masks.shape == (self.num_objs, img_size, img_size)
            assert ids.shape[0] == self.num_objs
