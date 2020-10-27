import torch
from mmdet.models import build_detector
from mmdet.models.detectors import BaseDetector

from mmtrack.core.motion import flow_warp_feats
from ..builder import MODELS, build_motion


@MODELS.register_module()
class DffTwoStage(BaseDetector):

    def __init__(self, detector, motion, train_cfg=None, test_cfg=None):
        super(DffTwoStage, self).__init__()
        self.detector = build_detector(detector, train_cfg, test_cfg)
        self.motion = build_motion(motion)
        self.motion.init_weights()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, img):
        return self.detector.extract_feat(img)

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

        flow_img = torch.cat((img, ref_img[:, 0]), dim=1)
        flow = self.motion(flow_img, img_metas)
        ref_x = self.extract_feat(ref_img[:, 0])
        x = []
        for i in range(len(ref_x)):
            x_single = flow_warp_feats(ref_x[i], flow)
            x.append(x_single)

        losses = dict()

        # RPN forward and loss
        if self.detector.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.detector.rpn_head.forward_train(
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

        return losses

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        key_frame_interval = self.test_cfg.get('key_frame_interval', 10)
        frame_id = img_metas[0]['frame_id']
        is_key_frame = False if frame_id % key_frame_interval else True

        if is_key_frame:
            x = self.extract_feat(img)
            self.key_img = img
            self.key_img_feats = x
        else:
            flow_img = torch.cat((img, self.key_img), dim=1)
            flow = self.motion(flow_img, img_metas)
            x = []
            for i in range(len(self.key_img_feats)):
                x_single = flow_warp_feats(self.key_img_feats[i], flow)
                x.append(x_single)

        if proposals is None:
            proposal_list = self.detector.rpn_head.simple_test_rpn(
                x, img_metas)
        else:
            proposal_list = proposals

        outs = self.detector.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)
        results = dict()
        results['bbox_results'] = outs[0]
        if len(outs) == 2:
            results['segm_results'] = outs[1]
        return results

    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        raise NotImplementedError
