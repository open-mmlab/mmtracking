import torch
from addict import Dict
from mmdet.core import bbox2result

from mmtrack.core.motion import flow_warp_feats
from ..builder import MODELS, build_detector, build_motion
from .base import BaseVideoDetector


@MODELS.register_module()
class DFF(BaseVideoDetector):
    """Deep Feature Flow for Video Recognition.

    This video object detector is the implementation of `DFF
    <https://arxiv.org/abs/1611.07715>`_.
    """

    def __init__(self,
                 detector,
                 motion,
                 pretrains=None,
                 frozen_modules=None,
                 train_cfg=None,
                 test_cfg=None):
        super(DFF, self).__init__()
        self.detector = build_detector(detector)
        self.motion = build_motion(motion)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrains)
        if frozen_modules is not None:
            self.freeze_module(frozen_modules)

    def init_weights(self, pretrain):
        """Initialize the weights of modules in video detector.

        Args:
            pretrained (dict): Path to pre-trained weights.
        """
        if pretrain is None:
            pretrain = dict()
        assert isinstance(pretrain, dict), '`pretrain` must be a dict.'
        if self.with_detector and pretrain.get('detector', False):
            self.init_module('detector', pretrain['detector'])
        if self.with_motion:
            self.init_module('motion', pretrain.get('motion', None))

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      ref_img,
                      ref_img_metas,
                      ref_gt_bboxes,
                      ref_gt_labels,
                      gt_instance_ids=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      ref_gt_instance_ids=None,
                      ref_gt_bboxes_ignore=None,
                      ref_gt_masks=None,
                      ref_proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmtrack/datasets/pipelines/formatting.py:VideoCollect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box.

            ref_img (Tensor): of shape (N, 1, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
                1 denotes there is only one ref image for each img.

            ref_img_metas (list[list[dict]]): The first list only has one
                element. The second list contains ref image info dict where
                each dict has: 'img_shape', 'scale_factor', 'flip', and may
                also contain 'filename', 'ori_shape', 'pad_shape', and
                'img_norm_cfg'. For details on the values of these keys see
                `mmtrack/datasets/pipelines/formatting.py:VideoCollect`.

            ref_gt_bboxes (list[Tensor]): The list only has one Tensor. The
                Tensor contains ground truth bboxes for each ref image with
                shape (num_all_ref_gts, 5) in
                [ref_img_id, tl_x, tl_y, br_x, br_y] format. The ref_img_id
                start from 0.

            ref_gt_labels (list[Tensor]): The list only has one Tensor. The
                Tensor contains class indices corresponding to each ref box
                with shape (num_all_ref_gts, 2) in [ref_img_id, class_indice].

            gt_instance_ids (None | list[Tensor]): specify the instance id for
                each ground truth bboxes.

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals (None | Tensor) : override rpn proposals with custom
                proposals. Use when `with_rpn` is False.

            ref_gt_instance_ids (None | list[Tensor]): specify the instance id
                for each ground truth bboxes of ref images.

            ref_gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes of ref images can be ignored when computing the loss.

            ref_gt_masks (None | Tensor) : true segmentation masks for each
                box of ref image used if the architecture supports a
                segmentation task.

            ref_proposals (None | Tensor) : override rpn proposals with custom
                proposals of ref image. Use when `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        assert len(img) == 1, \
            'Dff video detectors only support 1 batch size per gpu for now.'
        is_video_data = img_metas[0]['is_video_data']

        flow_img = torch.cat((img, ref_img[:, 0]), dim=1)
        flow = self.motion(flow_img, img_metas)
        ref_x = self.detector.extract_feat(ref_img[:, 0])
        x = []
        for i in range(len(ref_x)):
            x_single = flow_warp_feats(ref_x[i], flow)
            if not is_video_data:
                x_single = 0 * x_single + ref_x[i]
            x.append(x_single)

        losses = dict()

        # Two stage detector
        if hasattr(self.detector, 'roi_head'):
            # RPN forward and loss
            if self.detector.with_rpn:
                proposal_cfg = self.detector.train_cfg.get(
                    'rpn_proposal', self.detector.test_cfg.rpn)
                rpn_losses, proposal_list = \
                    self.detector.rpn_head.forward_train(
                        x,
                        img_metas,
                        gt_bboxes,
                        gt_labels=None,
                        gt_bboxes_ignore=gt_bboxes_ignore,
                        proposal_cfg=proposal_cfg)
                losses.update(rpn_losses)
            else:
                proposal_list = proposals

            roi_losses = self.detector.roi_head.forward_train(
                x, img_metas, proposal_list, gt_bboxes, gt_labels,
                gt_bboxes_ignore, gt_masks, **kwargs)
            losses.update(roi_losses)
        # Single stage detector
        elif hasattr(self.detector, 'bbox_head'):
            bbox_losses = self.detector.bbox_head.forward_train(
                x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore)
            losses.update(bbox_losses)
        else:
            raise TypeError('detector must has roi_head or bbox_head.')

        return losses

    def extract_feats(self, img, img_metas):
        """Extract features for img during testing.

        Args:
            img (Tensor): of shape (1, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmtrack/datasets/pipelines/formatting.py:VideoCollect`.

        Returns:
            list[Tensor]: Each level feature map of all images.
        """
        key_frame_interval = self.test_cfg.get('key_frame_interval', 10)
        frame_id = img_metas[0].get('frame_id', -1)
        assert frame_id >= 0
        is_key_frame = False if frame_id % key_frame_interval else True

        if is_key_frame:
            self.memo = Dict()
            self.memo.img = img
            x = self.detector.extract_feat(img)
            self.memo.feats = x
        else:
            flow_img = torch.cat((img, self.memo.img), dim=1)
            flow = self.motion(flow_img, img_metas)
            x = []
            for i in range(len(self.memo.feats)):
                x_single = flow_warp_feats(self.memo.feats[i], flow)
                x.append(x_single)
        return x

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation.

        Args:
            img (Tensor): of shape (1, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmtrack/datasets/pipelines/formatting.py:VideoCollect`.

            proposals (None | Tensor): Override rpn proposals with custom
                proposals. Use when `with_rpn` is False. Defaults to None.

            rescale (bool): If False, then returned bboxes and masks will fit
                the scale of img, otherwise, returned bboxes and masks
                will fit the scale of original image shape. Defaults to False.

        Returns:
            dict[str : list(ndarray)]: The detection results.
        """
        x = self.extract_feats(img, img_metas)

        # Two stage detector
        if hasattr(self.detector, 'roi_head'):
            if proposals is None:
                proposal_list = self.detector.rpn_head.simple_test_rpn(
                    x, img_metas)
            else:
                proposal_list = proposals

            outs = self.detector.roi_head.simple_test(
                x, proposal_list, img_metas, rescale=rescale)
        # Single stage detector
        elif hasattr(self.detector, 'bbox_head'):
            outs = self.bbox_head(x)
            bbox_list = self.bbox_head.get_bboxes(
                *outs, img_metas, rescale=rescale)
            # skip post-processing when exporting to ONNX
            if torch.onnx.is_in_onnx_export():
                return bbox_list

            outs = [
                bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
                for det_bboxes, det_labels in bbox_list
            ]
        else:
            raise TypeError('detector must has roi_head or bbox_head.')

        results = dict()
        results['bbox_results'] = outs[0]
        if len(outs) == 2:
            results['segm_results'] = outs[1]
        return results

    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        raise NotImplementedError
