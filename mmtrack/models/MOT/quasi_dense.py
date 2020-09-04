import torch
import torch.nn as nn

from ..builder import MODELS, TRACKERS, build_tracker
from mmdet.models import TwoStageDetector, build_head


@MODELS.register_module()
class QuasiDenseFasterRCNN(TwoStageDetector):

    def __init__(self, tracker=None, *args, **kwargs):
        kwargs['roi_head'].update(track_train_cfg=kwargs['train_cfg']['embed'])
        super().__init__(*args, **kwargs)
        # self.tracker = build_tracker(tracker)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_mids,
                      ref_img,
                      ref_img_metas,
                      ref_gt_bboxes,
                      ref_gt_labels,
                      ref_gt_mids,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      ref_gt_bboxes_ignore=None,
                      ref_gt_masks=None,
                      **kwargs):
        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
        rpn_losses, proposal_list = self.rpn_head.forward_train(
            x,
            img_metas,
            gt_bboxes,
            gt_labels=None,
            gt_bboxes_ignore=gt_bboxes_ignore,
            proposal_cfg=proposal_cfg)
        losses.update(rpn_losses)

        ref_x = self.extract_feat(ref_img)
        ref_proposals = self.rpn_head.simple_test_rpn(ref_x, ref_img_metas)

        roi_losses = self.roi_head.forward_train(
            x, img_metas, proposal_list, gt_bboxes, gt_labels, gt_mids, ref_x,
            ref_img_metas, ref_proposals, ref_gt_bboxes, ref_gt_labels,
            gt_bboxes_ignore, gt_masks, ref_gt_bboxes_ignore, **kwargs)
        losses.update(roi_losses)

        return losses
