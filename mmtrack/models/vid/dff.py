import torch
from mmdet.core import bbox2result
from mmdet.models import build_detector

from mmtrack.core.motion import flow_warp_feats
from ..builder import MODELS, build_motion
from .base import BaseVideoDetector


@MODELS.register_module()
class DFF(BaseVideoDetector):

    def __init__(self,
                 detector,
                 motion,
                 pretrains=None,
                 frozen_modules=None,
                 train_cfg=None,
                 test_cfg=None):
        super(DFF, self).__init__()
        self.detector = build_detector(detector, train_cfg, test_cfg)
        self.motion = build_motion(motion)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrains)
        if frozen_modules is not None:
            self.freeze_module(frozen_modules)

    def init_weights(self, pretrain):
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
                proposal_cfg = self.train_cfg.get('rpn_proposal',
                                                  self.test_cfg.rpn)
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

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        key_frame_interval = self.test_cfg.get('key_frame_interval', 10)
        frame_id = img_metas[0].get('frame_id', -1)
        assert frame_id >= 0
        is_key_frame = False if frame_id % key_frame_interval else True

        if is_key_frame:
            x = self.detector.extract_feat(img)
            self.key_img = img
            self.key_img_feats = x
        else:
            flow_img = torch.cat((img, self.key_img), dim=1)
            flow = self.motion(flow_img, img_metas)
            x = []
            for i in range(len(self.key_img_feats)):
                x_single = flow_warp_feats(self.key_img_feats[i], flow)
                x.append(x_single)

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
