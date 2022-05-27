# Copyright (c) OpenMMLab. All rights reserved.

import copy
from typing import List, Optional, Tuple, Union

import torch
from addict import Dict
from mmengine.data import InstanceData

from mmtrack.core import TrackDataSample
from mmtrack.core.motion import flow_warp_feats
from mmtrack.registry import MODELS
from .base import BaseVideoDetector


@MODELS.register_module()
class DFF(BaseVideoDetector):
    """Deep Feature Flow for Video Recognition.

    This video object detector is the implementation of `DFF
    <https://arxiv.org/abs/1611.07715>`_.
    """

    def __init__(self,
                 detector: dict,
                 motion: dict,
                 preprocess_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 frozen_modules: Optional[Union[List[str], Tuple[str],
                                                str]] = None):
        super().__init__(preprocess_cfg, init_cfg)
        self.detector = MODELS.build(detector)
        self.motion = MODELS.build(motion)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if frozen_modules is not None:
            self.freeze_module(frozen_modules)

    def forward_train(self, batch_inputs: dict,
                      batch_data_samples: List[TrackDataSample],
                      **kwargs) -> dict:
        """
        Args:
            batch_inputs (dict[Tensor]): of shape (N, T, C, H, W) encoding
                input images. Typically these should be mean centered and std
                scaled. The N denotes batch size and must be 1 in DFF method.
                The T denotes the number of key/reference frames.
                - img (Tensor) : The key images.
                - ref_img (Tensor): The reference images.
            batch_data_samples (list[:obj:`TrackDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        img = batch_inputs['img']
        assert img.dim() == 5, 'The img must be 5D Tensor (N, T, C, H, W).'
        assert img.size(0) == 1, \
            'Dff video detectors only support 1 batch size per gpu for now.'
        img = img[0]

        ref_img = batch_inputs['ref_img']
        assert ref_img.dim() == 5, 'The img must be 5D Tensor (N, T, C, H, W).'
        assert ref_img.size(0) == 1, \
            'Dff video detectors only support 1 batch size per gpu for now.'
        ref_img = ref_img[0]

        assert len(batch_data_samples) == 1, \
            'Dff video detectors only support 1 batch size per gpu for now.'
        metainfo = batch_data_samples[0].metainfo

        flow_img = torch.cat((img, ref_img), dim=1)
        flow = self.motion(flow_img, metainfo, self.preprocess_cfg)
        ref_x = self.detector.extract_feat(ref_img)
        x = []
        for i in range(len(ref_x)):
            x_single = flow_warp_feats(ref_x[i], flow)
            if not metainfo['is_video_data']:
                x_single = 0 * x_single + ref_x[i]
            x.append(x_single)

        losses = dict()

        # TODO: wait for mmdet
        # Two stage detector
        if hasattr(self.detector, 'roi_head'):
            # RPN forward and loss
            if self.detector.with_rpn:
                proposal_cfg = self.detector.train_cfg.get(
                    'rpn_proposal', self.detector.test_cfg.rpn)
                rpn_data_samples = copy.deepcopy(batch_data_samples)
                # set cat_id of gt_labels to 0 in RPN
                for data_sample in rpn_data_samples:
                    data_sample.gt_instances.labels = \
                        torch.zeros_like(data_sample.gt_instances.labels)

                rpn_losses, rpn_results_list = \
                    self.detector.rpn_head.forward_train(
                        x, rpn_data_samples, proposal_cfg=proposal_cfg,
                        **kwargs)
                # avoid get same name with roi_head loss
                keys = rpn_losses.keys()
                for key in keys:
                    if 'loss' in key and 'rpn' not in key:
                        rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
                losses.update(rpn_losses)
            else:
                # TODO: Need check with Fast R-CNN
                rpn_results_list = []
                for i in range(len(batch_data_samples)):
                    results = InstanceData()
                    results.bboxes = batch_data_samples[i].proposals
                    rpn_results_list.append(results)

            roi_losses = self.detector.roi_head.forward_train(
                x, rpn_results_list, batch_data_samples, **kwargs)
            losses.update(roi_losses)
        # Single stage detector
        elif hasattr(self.detector, 'bbox_head'):
            bbox_losses = self.detector.bbox_head.forward_train(
                x, batch_data_samples, **kwargs)
            losses.update(bbox_losses)
        else:
            raise TypeError('detector must has roi_head or bbox_head.')

        return losses

    def extract_feats(self, img: torch.Tensor,
                      meta_info: dict) -> torch.Tensor:
        """Extract features for `img` during testing.

        Args:
            img (Tensor): of shape (T, C, H, W) encoding input image.
                Typically these should be mean centered and std scaled.
                The T denotes the number of key images and usually is 1 in
                DFF method.
            metainfo (dict): image information dict where each
                dict has: 'img_shape', 'scale_factor', 'flip', and may also
                contain 'filename', 'ori_shape', 'pad_shape', and
                'img_norm_cfg'. For details on the values of these keys see
                `mmtrack/datasets/transforms/formatting.py:PackTrackInputs`.

        Returns:
            list[Tensor]: Multi level feature maps of `img`.
        """
        key_frame_interval = self.test_cfg.get('key_frame_interval', 10)
        frame_id = meta_info.get('frame_id', -1)
        assert frame_id >= 0
        is_key_frame = False if frame_id % key_frame_interval else True

        if is_key_frame:
            self.memo = Dict()
            self.memo.img = img
            x = self.detector.extract_feat(img)
            self.memo.feats = x
        else:
            flow_img = torch.cat((img, self.memo.img), dim=1)
            flow = self.motion(flow_img, meta_info, self.preprocess_cfg)
            x = []
            for i in range(len(self.memo.feats)):
                x_single = flow_warp_feats(self.memo.feats[i], flow)
                x.append(x_single)
        return x

    def simple_test(self,
                    batch_inputs: dict,
                    batch_data_samples: List[TrackDataSample],
                    rescale: bool = False):
        """Test without augmentation.

        Args:
            batch_inputs (dict[Tensor]): of shape (N, T, C, H, W) encoding
                input images. Typically these should be mean centered and std
                scaled. The N denotes batch size and must be 1 in DFF method.
                The T denotes the number of key/reference frames.
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
            'Dff video detectors only support 1 batch size per gpu for now.'
        img = img[0]

        assert len(batch_data_samples) == 1, \
            'Dff video detectors only support 1 batch size per gpu for now.'

        metainfo = batch_data_samples[0].metainfo
        x = self.extract_feats(img, metainfo)

        track_data_sample = copy.deepcopy(batch_data_samples[0])

        # TODO: wait for mmdet
        # Two stage detector
        if hasattr(self.detector, 'roi_head'):
            if not hasattr(batch_data_samples[0], 'proposals'):
                rpn_results_list = self.detector.rpn_head.simple_test(
                    x, [metainfo], rescale=False)
            else:
                # TODO: Need check with Fast R-CNN
                rpn_results_list = []
                for i in range(len(batch_data_samples)):
                    results = InstanceData()
                    results.bboxes = batch_data_samples[i].proposals
                    rpn_results_list.append(results)

            results_list = self.detector.roi_head.simple_test(
                x, rpn_results_list, [metainfo], rescale=rescale)
            # connvert to DetDataSample
            det_data_sample_list = self.detector.postprocess_result(
                results_list)
            assert len(results_list) == 1

            track_data_sample.pred_det_instances = \
                det_data_sample_list[0].pred_instances
        # Single stage detector
        elif hasattr(self.detector, 'bbox_head'):
            results_list = self.detector.bbox_head.simple_test(
                x, [metainfo], rescale=rescale)

            track_data_sample.pred_det_instances = results_list[0]
        else:
            raise TypeError('detector must has roi_head or bbox_head.')

        return [track_data_sample]
