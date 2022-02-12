import torch
from mmdet.core.bbox.demodata import random_boxes

from mmtrack.models import TRACKERS


class TestTaoTracker(object):

    @classmethod
    def setup_class(cls):
        cfg = dict(
            init_score_thr=0.0001,
            obj_score_thr=0.0001,
            match_score_thr=0.5,
            memo_frames=10,
            momentum_embed=0.8,
            momentum_obj_score=0.5,
            obj_score_diff_thr=1.0,
            distractor_nms_thr=0.3,
            distractor_score_thr=0.5,
            match_metric='bisoftmax',
            match_with_cosine=True)
        tracker = TRACKERS.get('TaoTracker')
        cls.tracker = tracker(**cfg)
        cls.num_objs = 5

    def test_update_memo(self):
        ids = torch.arange(self.num_objs)
        bboxes = random_boxes(self.num_objs, 64)
        labels = torch.arange(self.num_objs)
        embeds = torch.randn(self.num_objs, 256)

        self.tracker.update_memo(
            ids=ids, bboxes=bboxes, labels=labels, embeds=embeds, frame_id=0)

        for tid in range(self.num_objs):
            assert len(self.tracker.tracklets[tid]['bboxes']) == 1
            assert len(self.tracker.tracklets[tid]['labels']) == 1
            assert self.tracker.tracklets[tid]['embeds'].equal(
                embeds[tid]) == True  # noqa
            assert len(self.tracker.tracklets[tid]['frame_ids']) == 1

        ids = torch.tensor([self.num_objs - 1])
        bboxes = random_boxes(1, 64)
        labels = torch.tensor([self.num_objs])
        embeds = torch.randn(1, 256)
        new_embeds = (
            1 - self.tracker.momentum_embed) * self.tracker.tracklets[
                ids.item()]['embeds'] + self.tracker.momentum_embed * embeds

        self.tracker.update_memo(
            ids=ids, bboxes=bboxes, labels=labels, embeds=embeds, frame_id=1)

        assert self.tracker.tracklets[ids.item()]['embeds'].equal(
            new_embeds[0]) == True  # noqa

    def test_memo(self):
        memo_bboxes, memo_labels, memo_embeds, memo_ids = self.tracker.memo
        assert memo_bboxes.shape[0] == memo_labels.shape[0]
        assert memo_embeds.shape[0] == memo_labels.shape[0]
        assert memo_ids.shape[0] == memo_embeds.shape[0]

    def test_init_tracklets(self):
        ids = torch.tensor([-1, -1])
        obj_scores = torch.tensor([1e-6, 0.5])
        self.tracker.init_tracklets(ids, obj_scores)
        assert self.tracker.num_tracklets == 1

    def test_match(self):
        self.tracker.reset()
        bboxes = random_boxes(self.num_objs, 64)
        scores = torch.rand((self.num_objs, 1))
        bboxes = torch.cat((bboxes, scores), dim=1)
        embeds = torch.randn(self.num_objs, 256)

        labels = torch.arange(self.num_objs)

        for frame_id in range(3):
            bboxes, labels, ids = self.tracker.match(bboxes, labels, embeds,
                                                     frame_id)

            assert bboxes.shape[0] == labels.shape[0]
            assert bboxes.shape[0] == ids.shape[0]
