# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import MagicMock

import torch
from mmdet.core.bbox.demodata import random_boxes

from mmtrack.models import TRACKERS


class TestQuasiDenseTAOTracker(object):

    @classmethod
    def setup_class(cls):
        cfg = dict(
            init_score_thr=0.0001,
            obj_score_thr=0.0001,
            match_score_thr=0.5,
            memo_frames=10,
            memo_momentum=0.8,
            momentum_obj_score=0.5,
            obj_score_diff_thr=1.0,
            distractor_nms_thr=0.3,
            distractor_score_thr=0.5,
            match_metric='bisoftmax',
            match_with_cosine=True)
        tracker = TRACKERS.get('QuasiDenseTAOTracker')
        cls.tracker = tracker(**cfg)
        cls.num_objs = 5

    def test_update(self):
        ids = torch.arange(self.num_objs)
        bboxes = random_boxes(self.num_objs, 64)
        labels = torch.arange(self.num_objs)
        embeds = torch.randn(self.num_objs, 256)

        self.tracker.update(
            ids=ids, bboxes=bboxes, embeds=embeds, labels=labels, frame_id=0)

        for tid in range(self.num_objs):
            assert self.tracker.tracks[tid]['bboxes'][-1].equal(bboxes[tid])
            assert self.tracker.tracks[tid]['embeds'].equal(embeds[tid])
            assert self.tracker.tracks[tid]['labels'][-1].equal(labels[tid])

        ids = torch.tensor([self.num_objs - 1])
        bboxes = random_boxes(1, 64)
        labels = torch.tensor([self.num_objs])
        embeds = torch.randn(1, 256)
        new_embeds = (1 - self.tracker.memo_momentum) * self.tracker.tracks[
            ids.item()]['embeds'] + self.tracker.memo_momentum * embeds

        self.tracker.update(
            ids=ids, bboxes=bboxes, labels=labels, embeds=embeds, frame_id=1)

        assert self.tracker.tracks[ids.item()]['embeds'].equal(new_embeds[0])

    def test_memo(self):
        memo_bboxes, memo_labels, memo_embeds, memo_ids = self.tracker.memo  # noqa
        assert memo_bboxes.shape[0] == memo_labels.shape[0]
        assert memo_labels.shape[0] == memo_embeds.shape[0]
        assert memo_embeds.shape[0] == memo_ids.shape[0]
        assert memo_ids.shape[0] == memo_bboxes.shape[0]

    def test_track(self):
        self.tracker.reset()
        img_size, feats_channel = 64, 256
        img_metas = [dict(scale_factor=1.0)]

        model = MagicMock()
        model.track_head.extract_roi_feats = MagicMock(
            return_value=(torch.rand(self.num_objs, feats_channel, 7, 7),
                          [self.num_objs]))
        model.track_head.simple_test = MagicMock(
            return_value=torch.rand((self.num_objs, self.num_objs + 1)))

        feats = torch.rand((1, feats_channel, img_size, img_size))

        bboxes = random_boxes(self.num_objs, 64)
        scores = torch.rand((self.num_objs, 1))
        bboxes = torch.cat((bboxes, scores), dim=1)

        labels = torch.arange(self.num_objs)

        for frame_id in range(3):
            bboxes, labels, ids = self.tracker.track(img_metas, feats, model,
                                                     bboxes, labels, frame_id)

            assert bboxes.shape[0] == labels.shape[0]
            assert labels.shape[0] == labels.shape[0]
            assert ids.shape[0] == labels.shape[0]
