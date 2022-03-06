# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdet.core.bbox.demodata import random_boxes

from mmtrack.models import TRACKERS


class TestQuasiDenseEmbedTracker(object):

    @classmethod
    def setup_class(cls):
        cfg = dict(
            init_score_thr=0.9,
            obj_score_thr=0.5,
            match_score_thr=0.5,
            memo_tracklet_frames=30,
            memo_backdrop_frames=1,
            memo_momentum=0.8,
            nms_conf_thr=0.5,
            nms_backdrop_iou_thr=0.3,
            nms_class_iou_thr=0.7,
            with_cats=True,
            match_metric='bisoftmax')
        tracker = TRACKERS.get('QuasiDenseEmbedTracker')
        cls.tracker = tracker(**cfg)
        cls.num_objs = 5

    def test_update_memo(self):
        ids = torch.arange(self.num_objs)
        bboxes = random_boxes(self.num_objs, 64)
        labels = torch.arange(self.num_objs)
        embeds = torch.randn(self.num_objs, 256)

        self.tracker.update_memo(
            ids=ids, bboxes=bboxes, embeds=embeds, labels=labels, frame_id=0)

        for tid in range(self.num_objs):
            assert self.tracker.tracklets[tid]['bbox'].equal(bboxes[tid])
            assert self.tracker.tracklets[tid]['embed'].equal(embeds[tid])
            assert self.tracker.tracklets[tid]['label'].equal(labels[tid])
            assert self.tracker.tracklets[tid]['acc_frame'] == 0
            assert self.tracker.tracklets[tid]['last_frame'] == 0
            assert len(self.tracker.tracklets[tid]['velocity']) == len(
                bboxes[tid])

        ids = torch.tensor([self.num_objs - 1])
        bboxes = random_boxes(1, 64)
        labels = torch.tensor([self.num_objs])
        embeds = torch.randn(1, 256)
        new_embeds = (1 - self.tracker.memo_momentum) * self.tracker.tracklets[
            ids.item()]['embed'] + self.tracker.memo_momentum * embeds

        self.tracker.update_memo(
            ids=ids, bboxes=bboxes, labels=labels, embeds=embeds, frame_id=1)

        assert self.tracker.tracklets[ids.item()]['embed'].equal(
            new_embeds[0]) == True  # noqa

    def test_memo(self):
        memo_bboxes, memo_labels, memo_embeds, memo_ids, memo_vs = self.tracker.memo  # noqa
        assert memo_bboxes.shape[0] == memo_labels.shape[0]
        assert memo_embeds.shape[0] == memo_labels.shape[0]
        assert memo_ids.shape[0] == memo_vs.shape[0]
        assert memo_vs.shape[0] == memo_embeds.shape[0]

    def test_track(self):
        self.tracker.reset()
        bboxes = random_boxes(self.num_objs, 64)
        scores = torch.rand((self.num_objs, 1))
        bboxes = torch.cat((bboxes, scores), dim=1)
        embeds = torch.randn(self.num_objs, 256)

        labels = torch.arange(self.num_objs)

        for frame_id in range(3):
            bboxes, labels, ids = self.tracker.track(bboxes, labels, embeds,
                                                     frame_id)

            assert bboxes.shape[0] == labels.shape[0]
            assert bboxes.shape[0] == ids.shape[0]
