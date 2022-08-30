# Copyright (c) OpenMMLab. All rights reserved.

import copy
from typing import List, Optional, Tuple, Union

import torch
from addict import Dict
from mmengine.structures import InstanceData
from torch import Tensor

from mmtrack.registry import MODELS
from mmtrack.utils import OptConfigType, OptMultiConfig, SampleList
from ..task_modules.motion import flow_warp_feats
from .base import BaseVideoDetector


@MODELS.register_module()
class FGFA(BaseVideoDetector):
    """Flow-Guided Feature Aggregation for Video Object Detection.

    This video object detector is the implementation of `FGFA
    <https://arxiv.org/abs/1703.10025>`_.
    """

    def __init__(self,
                 detector: dict,
                 motion: dict,
                 aggregator: dict,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 frozen_modules: Optional[Union[List[str], Tuple[str],
                                                str]] = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(data_preprocessor, init_cfg)
        self.detector = MODELS.build(detector)
        self.motion = MODELS.build(motion)
        self.aggregator = MODELS.build(aggregator)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.preprocess_cfg = data_preprocessor

        if frozen_modules is not None:
            self.freeze_module(frozen_modules)

    def loss(self, inputs: dict, data_samples: SampleList, **kwargs) -> dict:
        """
        Args:
            inputs (dict[Tensor]): of shape (N, T, C, H, W) encoding
                input images. Typically these should be mean centered and std
                scaled. The N denotes batch size and must be 1 in FGFA method.
                The T denotes the number of key/reference frames.
                - img (Tensor) : The key images.
                - ref_img (Tensor): The reference images.
            data_samples (list[:obj:`TrackDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        img = inputs['img']
        assert img.dim() == 5, 'The img must be 5D Tensor (N, T, C, H, W).'
        assert img.size(0) == 1, \
            'FGFA video detectors only support 1 batch size per gpu for now.'
        assert img.size(1) == 1, \
            'FGFA video detector only has 1 key image per batch.'
        img = img[0]

        ref_img = inputs['ref_img']
        assert ref_img.dim() == 5, 'The img must be 5D Tensor (N, T, C, H, W).'
        assert ref_img.size(0) == 1, \
            'FGFA video detectors only support 1 batch size per gpu for now.'
        ref_img = ref_img[0]

        assert len(data_samples) == 1, \
            'FGFA video detectors only support 1 batch size per gpu for now.'
        metainfo = data_samples[0].metainfo

        num_ref_imgs = ref_img.size(0)
        flow_imgs = torch.cat((img.repeat(num_ref_imgs, 1, 1, 1), ref_img),
                              dim=1)
        flows = self.motion(flow_imgs, metainfo, self.preprocess_cfg)

        img_x = self.detector.extract_feat(img)
        ref_img_x = self.detector.extract_feat(ref_img)
        assert len(img_x) == len(ref_img_x)

        x = []
        for i in range(len(img_x)):
            ref_x_single = flow_warp_feats(ref_img_x[i], flows)
            agg_x_single = self.aggregator(img_x[i], ref_x_single)
            x.append(agg_x_single)

        losses = dict()

        # Two stage detector
        if hasattr(self.detector, 'roi_head'):
            # RPN forward and loss
            if self.detector.with_rpn:
                proposal_cfg = self.detector.train_cfg.get(
                    'rpn_proposal', self.detector.test_cfg.rpn)
                rpn_data_samples = copy.deepcopy(data_samples)
                # set cat_id of gt_labels to 0 in RPN
                for data_sample in rpn_data_samples:
                    data_sample.gt_instances.labels = \
                        torch.zeros_like(data_sample.gt_instances.labels)

                rpn_losses, rpn_results_list = \
                    self.detector.rpn_head.loss_and_predict(
                        x, rpn_data_samples, proposal_cfg=proposal_cfg,
                        **kwargs)
                # avoid get same name with roi_head loss
                keys = rpn_losses.keys()
                for key in keys:
                    if 'loss' in key and 'rpn' not in key:
                        rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
                losses.update(rpn_losses)
            else:
                rpn_results_list = []
                for i in range(len(data_samples)):
                    results = InstanceData()
                    results.bboxes = data_samples[i].proposals
                    rpn_results_list.append(results)

            roi_losses = self.detector.roi_head.loss(x, rpn_results_list,
                                                     data_samples, **kwargs)
            losses.update(roi_losses)
        # Single stage detector
        elif hasattr(self.detector, 'bbox_head'):
            bbox_losses = self.detector.bbox_head.loss(x, data_samples,
                                                       **kwargs)
            losses.update(bbox_losses)
        else:
            raise TypeError('detector must has roi_head or bbox_head.')

        return losses

    def extract_feats(self, img: Tensor, ref_img: Union[Tensor, None],
                      metainfo: dict) -> List[Tensor]:
        """Extract features for `img` during testing.

        Args:
            img (Tensor): of shape (T, C, H, W) encoding input image.
                Typically these should be mean centered and std scaled.
                The T denotes the number of key images and usually is 1 in
                FGFA method.
            ref_img (Tensor | None): of shape (T, C, H, W) encoding
                reference image. Typically these should be mean centered
                and std scaled. The T denotes the number of reference images.
                There may be no reference images in some cases.
            metainfo (dict): image information dict where each
                dict has: 'img_shape', 'scale_factor', 'flip', and may also
                contain 'filename', 'ori_shape', 'pad_shape', and
                'img_norm_cfg'. For details on the values of these keys see
                `mmtrack/datasets/transforms/formatting.py:PackTrackInputs`.

        Returns:
            list[Tensor]: Multi level feature maps of `img`.
        """
        frame_id = metainfo.get('frame_id', -1)
        assert frame_id >= 0
        num_left_ref_imgs = metainfo.get('num_left_ref_imgs', -1)
        frame_stride = metainfo.get('frame_stride', -1)

        # test with adaptive stride
        if frame_stride < 1:
            if frame_id == 0:
                self.memo = Dict()
                self.memo.img = ref_img
                ref_x = self.detector.extract_feat(ref_img)
                # 'tuple' object (e.g. the output of FPN) does not support
                # item assignment
                self.memo.feats = []
                for i in range(len(ref_x)):
                    self.memo.feats.append(ref_x[i])
            x = self.detector.extract_feat(img)
        # test with fixed stride
        else:
            if frame_id == 0:
                self.memo = Dict()
                self.memo.img = ref_img
                ref_x = self.detector.extract_feat(ref_img)
                # 'tuple' object (e.g. the output of FPN) does not support
                # item assignment
                self.memo.feats = []
                # the features of img is same as ref_x[i][[num_left_ref_imgs]]
                x = []
                for i in range(len(ref_x)):
                    self.memo.feats.append(ref_x[i])
                    x.append(ref_x[i][[num_left_ref_imgs]])
            elif frame_id % frame_stride == 0:
                assert ref_img is not None
                x = []
                ref_x = self.detector.extract_feat(ref_img)
                for i in range(len(ref_x)):
                    self.memo.feats[i] = torch.cat(
                        (self.memo.feats[i], ref_x[i]), dim=0)[1:]
                    x.append(self.memo.feats[i][[num_left_ref_imgs]])
                self.memo.img = torch.cat((self.memo.img, ref_img), dim=0)[1:]
            else:
                assert ref_img is None
                x = self.detector.extract_feat(img)

        flow_imgs = torch.cat(
            (img.repeat(self.memo.img.shape[0], 1, 1, 1), self.memo.img),
            dim=1)
        flows = self.motion(flow_imgs, metainfo, self.preprocess_cfg)

        agg_x = []
        for i in range(len(x)):
            agg_x_single = flow_warp_feats(self.memo.feats[i], flows)
            if frame_stride < 1:
                agg_x_single = torch.cat((x[i], agg_x_single), dim=0)
            else:
                agg_x_single[num_left_ref_imgs] = x[i]
            agg_x_single = self.aggregator(x[i], agg_x_single)
            agg_x.append(agg_x_single)
        return agg_x

    def predict(self,
                inputs: dict,
                data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Test without augmentation.

        Args:
            inputs (dict[Tensor]): of shape (N, T, C, H, W) encoding
                input images. Typically these should be mean centered and std
                scaled. The N denotes batch size and must be 1 in FGFA method.
                The T denotes the number of key/reference frames.
                - img (Tensor) : The key images.
                - ref_img (Tensor, Optional): The reference images.
            data_samples (list[:obj:`TrackDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance`.
            rescale (bool, Optional): If False, then returned bboxes and masks
                will fit the scale of img, otherwise, returned bboxes and masks
                will fit the scale of original image shape. Defaults to True.

        Returns:
            list[obj:`TrackDataSample`]: Tracking results of the
            input images. Each TrackDataSample usually contains
            ``pred_det_instances`` or ``pred_track_instances``.
        """
        img = inputs['img']
        assert img.dim() == 5, 'The img must be 5D Tensor (N, T, C, H, W).'
        assert img.size(0) == 1, \
            'FGFA video detectors only support 1 batch size per gpu for now.'
        assert img.size(1) == 1, \
            'FGFA video detector only has 1 key image per batch.'
        img = img[0]

        if 'ref_img' in inputs:
            ref_img = inputs['ref_img']
            assert ref_img.dim(
            ) == 5, 'The img must be 5D Tensor (N, T, C, H, W).'
            assert ref_img.size(0) == 1, 'FGFA video detectors only support' \
                                         ' 1 batch size per gpu for now.'
            ref_img = ref_img[0]
        else:
            ref_img = None

        assert len(data_samples) == 1, \
            'FGFA video detectors only support 1 batch size per gpu for now.'
        metainfo = data_samples[0].metainfo

        x = self.extract_feats(img, ref_img, metainfo)

        track_data_sample = copy.deepcopy(data_samples[0])

        # Two stage detector
        if hasattr(self.detector, 'roi_head'):
            if not hasattr(data_samples[0], 'proposals'):
                rpn_results_list = self.detector.rpn_head.predict(
                    x, data_samples, rescale=False)
            else:
                rpn_results_list = []
                for i in range(len(data_samples)):
                    results = InstanceData()
                    results.bboxes = data_samples[i].proposals
                    rpn_results_list.append(results)

            results_list = self.detector.roi_head.predict(
                x, rpn_results_list, data_samples, rescale=rescale)
            track_data_sample.pred_det_instances = results_list[0]
        # Single stage detector
        elif hasattr(self.detector, 'bbox_head'):
            results_list = self.detector.bbox_head.predict(
                x, data_samples, rescale=rescale)

            track_data_sample.pred_det_instances = results_list[0]
        else:
            raise TypeError('detector must has roi_head or bbox_head.')

        return [track_data_sample]
