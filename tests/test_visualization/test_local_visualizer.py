# Copyright (c) OpenMMLab. All rights reserved.
import os
from unittest import TestCase

import cv2
import numpy as np
import torch
from mmengine.structures import InstanceData

from mmtrack.structures import TrackDataSample
from mmtrack.visualization import DetLocalVisualizer, TrackLocalVisualizer


class TestTrackLocalVisualizer(TestCase):

    @staticmethod
    def _get_gt_instances():
        bboxes = np.array([[912, 484, 1009, 593], [1338, 418, 1505, 797]])
        masks = np.zeros((2, 1080, 1920), dtype=np.bool_)
        for i, bbox in enumerate(bboxes):
            masks[i, bbox[1]:bbox[3], bbox[0]:bbox[2]] = True
        instances_data = dict(
            bboxes=torch.tensor(bboxes),
            masks=masks,
            instances_id=torch.tensor([1, 2]),
            labels=torch.tensor([0, 1]))
        instances = InstanceData(**instances_data)
        return instances

    @staticmethod
    def _get_pred_instances():
        instances_data = dict(
            bboxes=torch.tensor([[900, 500, 1000, 600], [1300, 400, 1500,
                                                         800]]),
            instances_id=torch.tensor([1, 2]),
            labels=torch.tensor([0, 1]),
            scores=torch.tensor([0.955, 0.876]))
        instances = InstanceData(**instances_data)
        return instances

    @staticmethod
    def _assert_image_and_shape(out_file, out_shape):
        assert os.path.exists(out_file)
        drawn_img = cv2.imread(out_file)
        assert drawn_img.shape == out_shape
        os.remove(out_file)

    def test_add_datasample(self):
        out_file = 'out_file.jpg'
        h, w = 1080, 1920
        image = np.random.randint(0, 256, size=(h, w, 3)).astype('uint8')
        gt_instances = self._get_gt_instances()
        pred_instances = self._get_pred_instances()
        track_data_sample = TrackDataSample()
        track_data_sample.gt_instances = gt_instances
        track_data_sample.pred_track_instances = pred_instances

        track_local_visualizer = TrackLocalVisualizer(alpha=0.2)
        track_local_visualizer.dataset_meta = dict(
            classes=['pedestrian', 'vehicle'])

        # test gt_instances
        track_local_visualizer.add_datasample('image', image,
                                              track_data_sample, None)

        # test out_file
        track_local_visualizer.add_datasample(
            'image', image, track_data_sample, None, out_file=out_file)
        self._assert_image_and_shape(out_file, (h, w, 3))

        # test gt_instances and pred_instances
        track_local_visualizer.add_datasample(
            'image', image, track_data_sample, out_file=out_file)
        self._assert_image_and_shape(out_file, (h, 2 * w, 3))

        track_local_visualizer.add_datasample(
            'image',
            image,
            track_data_sample,
            draw_gt=False,
            out_file=out_file)
        self._assert_image_and_shape(out_file, (h, w, 3))

        track_local_visualizer.add_datasample(
            'image',
            image,
            track_data_sample,
            draw_pred=False,
            out_file=out_file)
        self._assert_image_and_shape(out_file, (h, w, 3))


class TestDetLocalVisualizer(TestCase):

    @staticmethod
    def _get_gt_instances():
        instances_data = dict(
            bboxes=np.array([[912, 484, 1009, 593], [1338, 418, 1505, 797]]),
            labels=torch.tensor([0, 1]),
            scores=torch.tensor([1., 1.]))
        instances = InstanceData(**instances_data)
        return instances

    @staticmethod
    def _get_pred_instances():
        instances_data = dict(
            bboxes=np.array([[900, 500, 1000, 600], [1300, 400, 1500, 800]]),
            labels=torch.tensor([0, 1]),
            scores=torch.tensor([0.955, 0.876]))
        instances = InstanceData(**instances_data)
        return instances

    @staticmethod
    def _assert_image_and_shape(out_file, out_shape):
        assert os.path.exists(out_file)
        drawn_img = cv2.imread(out_file)
        assert drawn_img.shape == out_shape
        os.remove(out_file)

    def test_add_datasample(self):
        out_file = 'out_file.jpg'
        h, w = 1080, 1920
        image = np.random.randint(0, 256, size=(h, w, 3)).astype('uint8')
        gt_instances = self._get_gt_instances()
        pred_instances = self._get_pred_instances()
        track_data_sample = TrackDataSample()
        track_data_sample.gt_instances = gt_instances
        track_data_sample.pred_det_instances = pred_instances

        det_local_visualizer = DetLocalVisualizer()
        det_local_visualizer.dataset_meta = dict(
            classes=['pedestrian', 'vehicle'])
        det_local_visualizer.add_datasample(
            'image', image, track_data_sample, out_file=out_file)
        self._assert_image_and_shape(out_file, (h, 2 * w, 3))
