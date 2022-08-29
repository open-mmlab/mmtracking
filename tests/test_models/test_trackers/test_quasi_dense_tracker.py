# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase
from unittest.mock import MagicMock

import torch

from mmtrack.registry import MODELS
from mmtrack.testing import demo_mm_inputs, random_boxes
from mmtrack.utils import register_all_modules


class TestQuasiDenseTracker(TestCase):

    @classmethod
    def setUpClass(cls):
        register_all_modules(init_default_scope=True)
        cfg = dict(
            type='QuasiDenseTracker',
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
            assert self.tracker.tracks[tid]['bbox'].equal(bboxes[tid])
            assert self.tracker.tracks[tid]['embed'].equal(embeds[tid])
            assert self.tracker.tracks[tid]['label'].equal(labels[tid])
            assert self.tracker.tracks[tid]['score'].equal(scores[tid])
            assert self.tracker.tracks[tid]['acc_frame'] == 0
            assert self.tracker.tracks[tid]['last_frame'] == 0
            assert len(self.tracker.tracks[tid]['velocity']) == len(
                bboxes[tid])

        ids = torch.tensor([self.num_objs - 1])
        bboxes = random_boxes(1, 64)
        labels = torch.tensor([self.num_objs])
        scores = torch.tensor([self.num_objs])
        embeds = torch.randn(1, 256)
        new_embeds = (1 - self.tracker.memo_momentum) * self.tracker.tracks[
            ids.item()]['embed'] + self.tracker.memo_momentum * embeds

        self.tracker.update(
            ids=ids,
            bboxes=bboxes,
            labels=labels,
            embeds=embeds,
            scores=scores,
            frame_id=1)

        assert self.tracker.tracks[ids.item()]['embed'].equal(
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
