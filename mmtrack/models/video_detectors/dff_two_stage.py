import torch
from mmdet.models import build_backbone
from mmdet.models.detectors import TwoStageDetector

from mmtrack.core import flow_warp_feats
from ..builder import MODELS


@MODELS.register_module()
class DffTwoStage(TwoStageDetector):

    def __init__(self,
                 backbone,
                 motion,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 key_frame_interval=10):
        super(DffTwoStage, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
        self.key_frame_interval = key_frame_interval
        self.motion = build_backbone(motion)
        self.motion.init_weights()

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
        if is_video_data:
            flow_img = torch.cat((img, ref_img[:, 0]), dim=1)
            flow = self.motion(flow_img, img_metas)

            ref_x = self.extract_feat(ref_img[:, 0])
            x = []
            for i in range(len(ref_x)):
                x_single = flow_warp_feats(flow, ref_x[i])
                x.append(x_single)
        else:
            x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        frame_id = img_metas[0]['frame_id']
        is_key_frame = False if frame_id % self.key_frame_interval else True

        if is_key_frame:
            x = self.extract_feat(img)
            self.key_img = img
            self.key_img_feats = x
        else:
            flow_img = torch.cat((img, self.key_img), dim=1)
            flow = self.motion(flow_img, img_metas)
            x = []
            for i in range(len(self.key_img_feats)):
                x_single = flow_warp_feats(flow, self.key_img_feats[i])
                x.append(x_single)

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        outs = self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)
        results = dict()
        if len(outs) == 1:
            results['bbox_result'] = outs[0]
        if len(outs) == 2:
            results['segm_result'] = outs[1]
        return results
