# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, Optional

from torch import Tensor

from mmtrack.models.mot import BaseMultiObjectTracker
from mmtrack.registry import MODELS
from mmtrack.utils import OptConfigType, OptMultiConfig, SampleList


@MODELS.register_module()
class MaskTrackRCNN(BaseMultiObjectTracker):
    """Video Instance Segmentation.

    This video instance segmentor is the implementation of`MaskTrack R-CNN
    <https://arxiv.org/abs/1905.04804>`_.

    Args:
        detector (dict): Configuration of detector. Defaults to None.
        track_head (dict): Configuration of track head. Defaults to None.
        tracker (dict): Configuration of tracker. Defaults to None.
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
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(data_preprocessor, init_cfg)

        if detector is not None:
            self.detector = MODELS.build(detector)
        assert hasattr(self.detector, 'roi_head'), \
            'MaskTrack R-CNN only supports two stage detectors.'

        if track_head is not None:
            self.track_head = MODELS.build(track_head)
        if tracker is not None:
            self.tracker = MODELS.build(tracker)

    def loss(self, inputs: Dict[str, Tensor], data_samples: SampleList,
             **kwargs) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Dict[str, Tensor]): of shape (N, T, C, H, W) encoding
                input images. Typically these should be mean centered and std
                scaled. The N denotes batch size.The T denotes the number of
                key/reference frames.
                - img (Tensor) : The key images.
                - ref_img (Tensor): The reference images.
            data_samples (list[:obj:`TrackDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance`.

        Returns:
            dict: A dictionary of loss components.
        """
        # modify the inputs shape to fit mmdet
        img = inputs['img']
        assert img.dim() == 5, 'The img must be 5D Tensor (N, T, C, H, W).'
        assert img.size(1) == 1, \
            'MaskTrackRCNN can only have 1 key frame and 1 reference frame.'
        img = img[:, 0]

        ref_img = inputs['ref_img']
        assert ref_img.dim() == 5, 'The img must be 5D Tensor (N, T, C, H, W).'
        assert ref_img.size(1) == 1, \
            'MaskTrackRCNN can only have 1 key frame and 1 reference frame.'
        ref_img = ref_img[:, 0]

        x = self.detector.extract_feat(img)
        ref_x = self.detector.extract_feat(ref_img)

        losses = dict()

        # RPN forward and loss
        if self.detector.with_rpn:
            proposal_cfg = self.detector.train_cfg.get(
                'rpn_proposal', self.detector.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(data_samples)
            rpn_losses, rpn_results_list = self.detector.rpn_head.\
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
        else:
            # TODO: Not support currently, should have a check at Fast R-CNN
            assert data_samples[0].get('proposals', None) is not None
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            rpn_results_list = [
                data_sample.proposals for data_sample in data_samples
            ]

        losses_detect = self.detector.roi_head.loss(x, rpn_results_list,
                                                    data_samples, **kwargs)
        losses.update(losses_detect)

        losses_track = self.track_head.loss(x, ref_x, rpn_results_list,
                                            data_samples, **kwargs)
        losses.update(losses_track)

        return losses

    def predict(self,
                inputs: dict,
                data_samples: SampleList,
                rescale: bool = True,
                **kwargs) -> SampleList:
        """Test without augmentation.

        Args:
            inputs (Dict[str, Tensor]): of shape (N, T, C, H, W)
                encoding input images. Typically these should be mean centered
                and std scaled. The N denotes batch size. The T denotes the
                number of key/reference frames.
                - img (Tensor) : The key images.
                - ref_img (Tensor): The reference images.
                In test mode, T = 1 and there is only ``img`` and no
                ``ref_img``.
            data_samples (list[:obj:`TrackDataSample`]): The batch
                data samples. It usually includes information such
                as ``gt_instances`` and 'metainfo'.
            rescale (bool, Optional): If False, then returned bboxes and masks
                will fit the scale of img, otherwise, returned bboxes and masks
                will fit the scale of original image shape. Defaults to True.

        Returns:
            SampleList: Tracking results of the input images.
            Each TrackDataSample usually contains ``pred_track_instances``.
        """
        img = inputs['img']
        assert img.dim() == 5, 'The img must be 5D Tensor (N, T, C, H, W).'
        assert img.size(1) == 1, \
            'MaskTrackRCNN can only have 1 key frame.'
        img = img[:, 0]

        assert len(data_samples) == 1, \
            'MaskTrackRCNN only support 1 batch size per gpu for now.'

        metainfo = data_samples[0].metainfo
        frame_id = metainfo.get('frame_id', -1)
        if frame_id == 0:
            self.tracker.reset()

        x = self.detector.extract_feat(img)

        rpn_results_list = self.detector.rpn_head.predict(x, data_samples)
        det_results = self.detector.roi_head.predict(
            x, rpn_results_list, data_samples, rescale=rescale)
        assert len(det_results) == 1, 'Batch inference is not supported.'
        assert 'masks' in det_results[0], 'There are no mask results.'
        track_data_sample = data_samples[0]
        track_data_sample.pred_det_instances = \
            det_results[0].clone()

        pred_track_instances = self.tracker.track(
            model=self,
            img=img,
            feats=x,
            data_sample=track_data_sample,
            rescale=rescale,
            **kwargs)
        track_data_sample.pred_track_instances = pred_track_instances

        return [track_data_sample]
