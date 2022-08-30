# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np
import pytest
import torch
from mmengine.structures import InstanceData

from mmtrack.structures import TrackDataSample


def _equal(a, b):
    if isinstance(a, (torch.Tensor, np.ndarray)):
        return (a == b).all()
    else:
        return a == b


class TestTrackDataSample(TestCase):

    def test_init(self):
        meta_info = dict(
            img_size=[256, 256],
            scale_factor=np.array([1.5, 1.5]),
            img_shape=torch.rand(4),
            ref_img_size=[512, 512, 2],
            ref_scale_factor=np.array([3, 3]),
            ref_img_shape=torch.rand(8))

        track_data_sample = TrackDataSample(metainfo=meta_info)
        assert 'img_size' in track_data_sample
        assert track_data_sample.img_size == [256, 256]
        assert track_data_sample.get('img_size') == [256, 256]

    def test_setter(self):
        track_data_sample = TrackDataSample()

        # test gt_instances
        gt_instances_data = dict(
            bboxes=torch.rand(4, 4),
            labels=torch.rand(4),
            masks=np.random.rand(4, 2, 2))
        gt_instances = InstanceData(**gt_instances_data)
        track_data_sample.gt_instances = gt_instances
        assert 'gt_instances' in track_data_sample
        assert _equal(track_data_sample.gt_instances.bboxes,
                      gt_instances_data['bboxes'])
        assert _equal(track_data_sample.gt_instances.labels,
                      gt_instances_data['labels'])
        assert _equal(track_data_sample.gt_instances.masks,
                      gt_instances_data['masks'])

        # test ignored_instances
        ignored_instances_data = dict(
            bboxes=torch.rand(4, 4), labels=torch.rand(4))
        ignored_instances = InstanceData(**ignored_instances_data)
        track_data_sample.ignored_instances = ignored_instances
        assert 'ignored_instances' in track_data_sample
        assert _equal(track_data_sample.ignored_instances.bboxes,
                      ignored_instances_data['bboxes'])
        assert _equal(track_data_sample.ignored_instances.labels,
                      ignored_instances_data['labels'])

        # test proposals
        proposals_data = dict(bboxes=torch.rand(4, 4), labels=torch.rand(4))
        proposals = InstanceData(**proposals_data)
        track_data_sample.proposals = proposals
        assert 'proposals' in track_data_sample
        assert _equal(track_data_sample.proposals.bboxes,
                      proposals_data['bboxes'])
        assert _equal(track_data_sample.proposals.labels,
                      proposals_data['labels'])

        # test pred_det_instances
        pred_det_instances_data = dict(
            bboxes=torch.rand(2, 4),
            labels=torch.rand(2),
            masks=np.random.rand(2, 2, 2))
        pred_det_instances = InstanceData(**pred_det_instances_data)
        track_data_sample.pred_det_instances = pred_det_instances
        assert 'pred_det_instances' in track_data_sample
        assert _equal(track_data_sample.pred_det_instances.bboxes,
                      pred_det_instances_data['bboxes'])
        assert _equal(track_data_sample.pred_det_instances.labels,
                      pred_det_instances_data['labels'])
        assert _equal(track_data_sample.pred_det_instances.masks,
                      pred_det_instances_data['masks'])

        # test pred_track_instances
        pred_track_instances_data = dict(
            bboxes=torch.rand(2, 4),
            labels=torch.rand(2),
            masks=np.random.rand(2, 2, 2))
        pred_track_instances = InstanceData(**pred_track_instances_data)
        track_data_sample.pred_track_instances = pred_track_instances
        assert 'pred_track_instances' in track_data_sample
        assert _equal(track_data_sample.pred_track_instances.bboxes,
                      pred_track_instances_data['bboxes'])
        assert _equal(track_data_sample.pred_track_instances.labels,
                      pred_track_instances_data['labels'])
        assert _equal(track_data_sample.pred_track_instances.masks,
                      pred_track_instances_data['masks'])

        # test type error
        with pytest.raises(AssertionError):
            track_data_sample.pred_det_instances = torch.rand(2, 4)
        with pytest.raises(AssertionError):
            track_data_sample.pred_track_instances = torch.rand(2, 4)

    def test_deleter(self):
        gt_instances_data = dict(
            bboxes=torch.rand(4, 4),
            labels=torch.rand(4),
            masks=np.random.rand(4, 2, 2))

        track_data_sample = TrackDataSample()
        gt_instances = InstanceData(data=gt_instances_data)
        track_data_sample.gt_instances = gt_instances
        assert 'gt_instances' in track_data_sample
        del track_data_sample.gt_instances
        assert 'gt_instances' not in track_data_sample
