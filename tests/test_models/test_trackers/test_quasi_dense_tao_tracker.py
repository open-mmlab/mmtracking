# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase
from unittest.mock import MagicMock

import torch

from mmtrack.registry import MODELS
from mmtrack.testing import demo_mm_inputs, random_boxes
from mmtrack.utils import register_all_modules


class TestQuasiDenseTAOTracker(TestCase):

    @classmethod
    def setUpClass(cls):
        register_all_modules(init_default_scope=True)
        cfg = dict(
            type='QuasiDenseTAOTracker',
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
        cls.tracker = MODELS.build(cfg)
        cls.num_objs = 5

    def test_update(self):
        ids = torch.arange(self.num_objs)
        bboxes = random_boxes(self.num_objs, 64)
        labels = torch.arange(self.num_objs)
        scores = torch.arange(self.num_objs)
        embeds = torch.randn(self.num_objs, 256)

        self.tracker.update(
            ids=ids,
            bboxes=bboxes,
            labels=labels,
            embeds=embeds,
            scores=scores,
            frame_id=0)

        for tid in range(self.num_objs):
            assert self.tracker.tracks[tid]['bboxes'][0].equal(
                bboxes[tid]) is True
            assert self.tracker.tracks[tid]['labels'][0].equal(
                labels[tid]) is True
            assert self.tracker.tracks[tid]['scores'][0].equal(
                scores[tid]) is True
            assert self.tracker.tracks[tid]['embeds'].equal(
                embeds[tid]) is True

        ids = torch.tensor([self.num_objs - 1])
        bboxes = random_boxes(1, 64)
        labels = torch.tensor([self.num_objs])
        scores = torch.tensor([self.num_objs])
        embeds = torch.randn(1, 256)
        new_embeds = (1 - self.tracker.memo_momentum) * self.tracker.tracks[
            ids.item()]['embeds'] + self.tracker.memo_momentum * embeds

        self.tracker.update(
            ids=ids,
            bboxes=bboxes,
            labels=labels,
            embeds=embeds,
            scores=scores,
            frame_id=1)

        assert self.tracker.tracks[ids.item()]['embeds'].equal(
            new_embeds[0]) == True  # noqa

    def test_track(self):
        img_size = 64
        img = torch.rand((1, 3, img_size, img_size))
        feats = torch.rand((1, 256, img_size, img_size))

        model = MagicMock()
        for frame_id in range(3):
            packed_inputs = demo_mm_inputs(
                batch_size=1, frame_id=0, num_ref_imgs=0)
            data_sample = packed_inputs['data_samples'][0]
            data_sample.pred_det_instances = data_sample.gt_instances.clone()
            # add fake scores
            scores = torch.ones(5)
            data_sample.pred_det_instances.scores = torch.FloatTensor(scores)
            pred_track_instances = self.tracker.track(
                model=model,
                img=img,
                feats=feats,
                data_sample=packed_inputs['data_samples'][0])

            bboxes = pred_track_instances.bboxes
            labels = pred_track_instances.labels
            ids = pred_track_instances.instances_id

            assert bboxes.shape[1] == 4
            assert bboxes.shape[0] == labels.shape[0]
            assert bboxes.shape[0] == ids.shape[0]
