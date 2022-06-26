# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, Optional

from torch import Tensor

from mmtrack.core import TrackDataSample
from mmtrack.core.utils import OptConfigType, OptMultiConfig, SampleList
from mmtrack.registry import MODELS
from .base import BaseMultiObjectTracker


@MODELS.register_module()
class QDTrack(BaseMultiObjectTracker):
    """Quasi-Dense Similarity Learning for Multiple Object Tracking.

    This multi object tracker is the implementation of `QDTrack
    <https://arxiv.org/abs/2006.06664>`_.

    Args:
        detector (dict): Configuration of detector. Defaults to None.
        track_head (dict): Configuration of track head. Defaults to None.
        tracker (dict): Configuration of tracker. Defaults to None.
        freeze_detector (bool): If True, freeze the detector weights.
            Defaults to False.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
           config of :class:`TrackDataPreprocessor`.  it usually includes,
            ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or list[dict]): Configuration of initialization.
            Defaults to None.
    """

    def __init__(self,
                 detector: Optional[dict] = None,
                 track_head: Optional[dict] = None,
                 tracker: Optional[dict] = None,
                 freeze_detector: bool = False,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(data_preprocessor, init_cfg)
        if detector is not None:
            self.detector = MODELS.build(detector)

        if track_head is not None:
            self.track_head = MODELS.build(track_head)

        if tracker is not None:
            self.tracker = MODELS.build(tracker)

        self.freeze_detector = freeze_detector
        if self.freeze_detector:
            self.freeze_module('detector')

    def loss(self, batch_inputs: Dict[str, Tensor],
             batch_data_samples: SampleList, **kwargs) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Dict[str, Tensor]): of shape (N, T, C, H, W) encoding
                input images. Typically these should be mean centered and std
                scaled. The N denotes batch size.The T denotes the number of
                key/reference frames.
                - img (Tensor) : The key images.
                - ref_img (Tensor): The reference images.
            batch_data_samples (list[:obj:`TrackDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance`.

        Returns:
            dict: A dictionary of loss components.
        """
        # modify the inputs shape to fit mmdet
        img = batch_inputs['img']
        assert img.dim() == 5, 'The img must be 5D Tensor (N, T, C, H, W).'
        assert img.size(1) == 1, \
            'QDTrack can only have 1 key frame and 1 reference frame.'
        img = img[:, 0]

        ref_img = batch_inputs['ref_img']
        assert ref_img.dim() == 5, 'The img must be 5D Tensor (N, T, C, H, W).'
        assert ref_img.size(1) == 1, \
            'QDTrack can only have 1 key frame and 1 reference frame.'
        ref_img = ref_img[:, 0]

        x = self.detector.extract_feat(img)
        ref_x = self.detector.extract_feat(ref_img)

        losses = dict()

        # RPN forward and loss
        assert self.detector.with_rpn, \
            'QDTrack only support detector with RPN.'

        proposal_cfg = self.detector.train_cfg.get('rpn_proposal',
                                                   self.detector.test_cfg.rpn)
        rpn_data_samples = copy.deepcopy(batch_data_samples)
        rpn_losses, rpn_results_list = self.detector.rpn_head. \
            loss_and_predict(x,
                             rpn_data_samples,
                             proposal_cfg=proposal_cfg,
                             **kwargs)
        # avoid get same name with roi_head loss
        keys = rpn_losses.keys()
        for key in keys:
            if 'loss' in key and 'rpn' not in key:
                rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
        losses.update(rpn_losses)

        losses_detect = self.detector.roi_head.loss(x, rpn_results_list,
                                                    batch_data_samples,
                                                    **kwargs)
        losses.update(losses_detect)

        # adjust the key of ref_img in batch_data_samples
        ref_rpn_data_samples = []
        for data_sample in batch_data_samples:
            ref_rpn_data_sample = TrackDataSample()
            ref_rpn_data_sample.set_metainfo(
                metainfo=dict(
                    img_shape=data_sample.metainfo['ref_img_shape'],
                    scale_factor=data_sample.metainfo['ref_scale_factor']))
            ref_rpn_data_samples.append(ref_rpn_data_sample)
        ref_rpn_results_list = self.detector.rpn_head.predict(
            ref_x, ref_rpn_data_samples, **kwargs)
        losses_track = self.track_head.loss(x, ref_x, rpn_results_list,
                                            ref_rpn_results_list,
                                            batch_data_samples, **kwargs)
        losses.update(losses_track)

        return losses

    def predict(self,
                batch_inputs: Dict[str, Tensor],
                batch_data_samples: SampleList,
                rescale: bool = False,
                **kwargs) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Dict[str, Tensor]): of shape (N, T, C, H, W) encoding
                input images. Typically these should be mean centered and std
                scaled. The N denotes batch size.The T denotes the number of
                key/reference frames.
                - img (Tensor) : The key images.
                - ref_img (Tensor): The reference images.
            batch_data_samples (list[:obj:`TrackDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance`.
            rescale (bool, Optional): If False, then returned bboxes and masks
                will fit the scale of img, otherwise, returned bboxes and masks
                will fit the scale of original image shape. Defaults to True.

        Returns:
            SampleList: Tracking results of the input images.
            Each TrackDataSample usually contains ``pred_det_instances``
            or ``pred_track_instances``.
        """
        img = batch_inputs['img']
        assert img.dim() == 5, 'The img must be 5D Tensor (N, T, C, H, W).'
        assert img.size(1) == 1, \
            'QDTrack can only have 1 key frame.'
        img = img[:, 0]

        assert len(batch_data_samples) == 1, \
            'QDTrack only support 1 batch size per gpu for now.'
        metainfo = batch_data_samples[0].metainfo
        frame_id = metainfo.get('frame_id', -1)
        if frame_id == 0:
            self.tracker.reset()

        x = self.detector.extract_feat(img)
        rpn_results_list = self.detector.rpn_head.predict(
            x, batch_data_samples)
        det_results = self.detector.roi_head.predict(
            x, rpn_results_list, batch_data_samples, rescale=rescale)

        track_data_sample = batch_data_samples[0]
        track_data_sample.pred_det_instances = \
            det_results[0].clone()

        track_data_sample = self.tracker.track(
            model=self,
            img=img,
            feats=x,
            data_sample=track_data_sample,
            rescale=rescale,
            **kwargs)

        return [track_data_sample]
