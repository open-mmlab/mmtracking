# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdet.core.bbox.demodata import random_boxes

from mmtrack.models import TRACKERS


class TestBaseTracker(object):

    @classmethod
    def setup_class(cls):
        cfg = dict(
            obj_score_thr=0.3,
            regression=dict(
                obj_score_thr=0.5,
                nms=dict(type='nms', iou_threshold=0.6),
                match_iou_thr=0.3),
            reid=dict(
                num_samples=10,
                img_scale=(256, 128),
                img_norm_cfg=None,
                match_score_thr=2.0,
                match_iou_thr=0.2),
            momentums=dict(embeds=0.5),
            num_frames_retain=5)
        tracker = TRACKERS.get('TracktorTracker')
        cls.tracker = tracker(**cfg)
        cls.momentums = cfg['momentums']
        cls.num_frames_retain = cfg['num_frames_retain']
        cls.num_objs = 5

    def test_init(self):
        bboxes = random_boxes(self.num_objs, 512)
        labels = torch.zeros(self.num_objs)
        embeds = torch.randn(self.num_objs, 256)
        ids = torch.arange(self.num_objs)
        self.tracker.update(
            ids=ids, bboxes=bboxes, labels=labels, embeds=embeds, frame_ids=0)

        assert self.tracker.ids == list(ids)
        assert self.tracker.memo_items == [
            'ids', 'bboxes', 'labels', 'embeds', 'frame_ids'
        ]
        for k, v in self.tracker.tracks[0].items():
            if k in self.momentums:
                assert isinstance(v, torch.Tensor)
            else:
                assert isinstance(v, list)

    def test_update(self):
        for i in range(1, self.num_frames_retain * 2):
            bboxes = random_boxes(self.num_objs, 512)
            labels = torch.zeros(self.num_objs, dtype=torch.int)
            embeds = torch.randn(self.num_objs, 256)
            ids = torch.arange(self.num_objs) + i
            self.tracker.update(
                ids=ids,
                bboxes=bboxes,
                labels=labels,
                embeds=embeds,
                frame_ids=i)
            if i < self.num_frames_retain:
                assert 0 in self.tracker.tracks
            else:
                assert 0 not in self.tracker.tracks

    def test_memo(self):
        memo = self.tracker.memo
        num_tracks = self.num_frames_retain * 2 - 1
        assert (memo.ids == torch.arange(
            self.num_frames_retain, self.num_frames_retain * 3 - 1)).all()
        assert memo.bboxes.shape[0] == num_tracks

    def test_get(self):
        ids = [self.num_frames_retain + 1, self.num_frames_retain + 2]

        bboxes = self.tracker.get('bboxes', ids)
        assert bboxes.shape == (2, 4)

        bboxes = self.tracker.get('bboxes', ids, num_samples=2)
        assert bboxes.shape == (2, 2, 4)

        bboxes = self.tracker.get(
            'bboxes', ids, num_samples=2, behavior='mean')
        assert bboxes.shape == (2, 4)
