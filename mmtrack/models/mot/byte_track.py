# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch

from mmtrack.core import TrackDataSample
from mmtrack.registry import MODELS
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
                 detector: Optional[dict] = None,
                 tracker: Optional[dict] = None,
                 motion: Optional[dict] = None,
                 preprocess_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        super().__init__(preprocess_cfg, init_cfg)

        if detector is not None:
            self.detector = MODELS.build(detector)

        if motion is not None:
            self.motion = MODELS.build(motion)

        if tracker is not None:
            self.tracker = MODELS.build(tracker)

    def forward_train(self, batch_inputs: dict,
                      batch_data_samples: List[TrackDataSample],
                      **kwargs) -> dict:
        """Forward function during training.
        Args:
            batch_inputs (dict[Tensor]): of shape (N, T, C, H, W) encoding
                input images. Typically these should be mean centered and std
                scaled. The N denotes batch size.The T denotes the number of
                key/reference frames.
                - img (Tensor) : The key images.
                - ref_img (Tensor): The reference images.
            batch_data_samples (list[:obj:`TrackDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # modify the inputs shape to fit mmdet
        img = batch_inputs['img']
        assert img.size(1) == 1
        # convert batch_inputs' shape to (N, C, H, W)
        img = torch.squeeze(img, dim=1)

        return self.detector.forward_train(img, batch_data_samples, **kwargs)

    def simple_test(self,
                    batch_inputs: dict,
                    batch_data_samples: List[TrackDataSample],
                    rescale: bool = False,
                    **kwargs):
        """Test without augmentation.

        Args:
            batch_inputs (dict[Tensor]): of shape (N, T, C, H, W) encoding
                input images. Typically these should be mean centered and std
                scaled. The N denotes batch size and must be 1 in ByteTrack
                method.The T denotes the number of key/reference frames.
                - img (Tensor) : The key images.
                - ref_img (Tensor, Optional): The reference images.
            batch_data_samples (list[:obj:`TrackDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance`.
            rescale (bool, Optional): If False, then returned bboxes and masks
                will fit the scale of img, otherwise, returned bboxes and masks
                will fit the scale of original image shape. Defaults to False.

        Returns:
            list[obj:`TrackDataSample`]: Tracking results of the
            input images. Each TrackDataSample usually contains
            ``pred_det_instances`` or ``pred_track_instances``.
        """
        img = batch_inputs['img']
        assert img.dim() == 5, 'The img must be 5D Tensor (N, T, C, H, W).'
        assert img.size(0) == 1, \
            'Bytetrack inference only support 1 batch size per gpu for now.'
        img = img[0]

        assert len(batch_data_samples) == 1, \
            'Bytetrack inference only support 1 batch size per gpu for now.'

        track_data_sample = batch_data_samples[0]
        metainfo = track_data_sample.metainfo

        det_results = self.detector.simple_test(
            img, [metainfo], rescale=rescale)
        assert len(det_results) == 1, 'Batch inference is not supported.'
        det_results = det_results[0]
        track_data_sample.pred_det_instances = \
            det_results.pred_instances.clone()

        track_data_sample = self.tracker(
            model=self,
            img=img,
            feats=None,
            data_sample=track_data_sample,
            **kwargs)

        return [track_data_sample]
