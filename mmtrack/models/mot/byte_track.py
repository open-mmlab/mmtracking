# Copyright (c) OpenMMLab. All rights reserved.

import torch
from mmdet.models import build_detector

from mmtrack.core import outs2results, results2outs
from ..builder import MODELS, build_motion, build_tracker
from .base import BaseMultiObjectTracker


@MODELS.register_module()
class ByteTrack(BaseMultiObjectTracker):
    """ByteTrack: Multi-Object Tracking by Associating Every Detection Box.

    This multi object tracker is the implementation of `ByteTrack
    <https://arxiv.org/abs/2110.06864>`_.

    Args:
        detector (dict): Configuration of detector. Defaults to None.
        tracker (dict): Configuration of tracker. Defaults to None.
        motion (dict): Configuration of motion. Defaults to None.
        init_cfg (dict): Configuration of initialization. Defaults to None.
    """

    def __init__(self,
                 detector=None,
                 tracker=None,
                 motion=None,
                 init_cfg=None):
        super().__init__(init_cfg)

        if detector is not None:
            self.detector = build_detector(detector)

        if motion is not None:
            self.motion = build_motion(motion)

        if tracker is not None:
            self.tracker = build_tracker(tracker)

    def forward_train(self, *args, **kwargs):
        """Forward function during training."""
        return self.detector.forward_train(*args, **kwargs)

    def simple_test(self, img, img_metas, rescale=False, **kwargs):
        """Test without augmentations.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            rescale (bool, optional): If False, then returned bboxes and masks
                will fit the scale of img, otherwise, returned bboxes and masks
                will fit the scale of original image shape. Defaults to False.

        Returns:
            dict[str : list(ndarray)]: The tracking results.
        """
        frame_id = img_metas[0].get('frame_id', -1)
        if frame_id == 0:
            self.tracker.reset()

        det_results = self.detector.simple_test(
            img, img_metas, rescale=rescale)
        assert len(det_results) == 1, 'Batch inference is not supported.'
        bbox_results = det_results[0]
        num_classes = len(bbox_results)

        outs_det = results2outs(bbox_results=bbox_results)
        det_bboxes = torch.from_numpy(outs_det['bboxes']).to(img)
        det_labels = torch.from_numpy(outs_det['labels']).to(img).long()

        track_bboxes, track_labels, track_ids = self.tracker.track(
            img=img,
            img_metas=img_metas,
            model=self,
            bboxes=det_bboxes,
            labels=det_labels,
            frame_id=frame_id,
            rescale=rescale,
            **kwargs)

        track_results = outs2results(
            bboxes=track_bboxes,
            labels=track_labels,
            ids=track_ids,
            num_classes=num_classes)
        det_results = outs2results(
            bboxes=det_bboxes, labels=det_labels, num_classes=num_classes)

        return dict(
            det_bboxes=det_results['bbox_results'],
            track_bboxes=track_results['bbox_results'])
