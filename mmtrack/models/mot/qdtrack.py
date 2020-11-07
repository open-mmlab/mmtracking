import os
import os.path as osp

import mmcv
import torch
from mmdet.core import bbox2result
from mmdet.models.builder import build_head

from mmtrack.core import track2result
from ..builder import MODELS, build_detector, build_tracker
from .base import BaseMultiObjectTracker


@MODELS.register_module()
class QDTrack(BaseMultiObjectTracker):

    def __init__(self,
                 detector=None,
                 track_head=None,
                 tracker=None,
                 pretrains=None,
                 frozen_modules=False,
                 dense_matching=True):
        super().__init__()
        if detector is not None:
            self.detector = build_detector(detector)

        if track_head is not None:
            self.track_head = build_head(track_head)

        if tracker is not None:
            self.tracker = build_tracker(tracker)

        self.dense_matching = dense_matching

        self.init_weights(pretrains)
        if frozen_modules is not None:
            self.freeze_module(frozen_modules)

    def init_weights(self, pretrain):
        if pretrain is None:
            pretrain = dict()
        assert isinstance(pretrain, dict), '`pretrain` must be a dict.'
        if self.with_detector and pretrain.get('detector', False):
            self.init_module('detector', pretrain['detector'])
        if self.with_track_head:
            self.init_module('track_head', pretrain.get('track_head', None))

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_match_indices,
                      ref_img,
                      ref_img_metas,
                      ref_gt_bboxes,
                      ref_gt_labels,
                      ref_gt_match_indices,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      ref_gt_bboxes_ignore=None,
                      ref_gt_masks=None,
                      **kwargs):
        losses = dict()
        # feature extraction
        x = self.detector.extract_feat(img)
        ref_x = self.detector.extract_feat(ref_img)

        # detector
        if hasattr(self.detector, 'roi_head'):
            # Two stage detector
            proposal_cfg = self.detector.train_cfg.get(
                'rpn_proposal', self.detector.test_cfg.rpn)
            rpn_losses, proposal_list = self.detector.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
            loss_bbox = self.detector.roi_head.forward_train(
                x, img_metas, proposal_list, gt_bboxes, gt_labels,
                gt_bboxes_ignore, gt_masks, **kwargs)
            if self.dense_matching:
                key_proposals = proposal_list.copy()
                ref_proposals = self.detector.rpn_head.simple_test_rpn(
                    ref_x, ref_img_metas)
            else:
                key_proposals = gt_bboxes.copy()
                ref_proposals = ref_proposals.copy()
            loss_track = self.track_head.forward_train(
                x, img_metas, key_proposals, gt_bboxes, gt_labels,
                gt_match_indices, ref_x, ref_img_metas, ref_proposals,
                ref_gt_bboxes, ref_gt_labels, ref_gt_match_indices,
                gt_bboxes_ignore, ref_gt_bboxes_ignore)
        elif hasattr(self.detector, 'bbox_head'):
            # Single stage detector
            loss_bbox = self.detector.bbox_head.forward_train(
                x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore)
            raise NotImplementedError()
        else:
            raise TypeError('detector must has roi_head or bbox_head.')
        losses.update(loss_bbox)
        losses.update(loss_track)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        frame_id = img_metas[0].get('frame_id', -1)
        if frame_id == 0:
            self.tracker.reset()

        x = self.detector.extract_feat(img)
        if hasattr(self.detector, 'roi_head'):
            proposals = self.detector.rpn_head.simple_test_rpn(x, img_metas)
            det_bboxes, det_labels = self.detector.roi_head.simple_test_bboxes(
                x,
                img_metas,
                proposals,
                self.detector.roi_head.test_cfg,
                rescale=rescale)
            # TODO: support batch inference
            det_bboxes = det_bboxes[0]
            det_labels = det_labels[0]
            num_classes = self.detector.roi_head.bbox_head.num_classes
        elif hasattr(self.detector, 'bbox_head'):
            num_classes = self.detector.bbox_head.num_classes
            raise NotImplementedError()
        else:
            raise TypeError('detector must has roi_head or bbox_head.')

        if frame_id < 0:
            bbox_result = bbox2result(det_bboxes, det_labels, num_classes)
            return dict(bbox_results=bbox_result, track_results=None)

        embeds = self.track_head.simple_test(x, img_metas, det_bboxes, rescale)

        if hasattr(self, 'save_variables') and self.save_variables is not None:
            save = dict()
            for k in self.save_variables:
                v = eval(k)
                if isinstance(v, torch.Tensor):
                    v = v.cpu()
                save[k] = v
            out_name = img_metas[0]['ori_filename'].rsplit('.', 1)[0] + '.pkl'
            out_path = osp.join(self.out_path, 'pkls', out_name)
            os.makedirs(out_path.rsplit('/', 1)[0], exist_ok=True)
            mmcv.dump(save, out_path)

        bboxes, labels, ids = self.tracker.match(
            bboxes=det_bboxes,
            labels=det_labels,
            embeds=embeds,
            frame_id=frame_id,
            temperature=self.track_head.embed_head.softmax_temperature)
        track_result = track2result(bboxes, labels, ids, num_classes)
        bbox_result = bbox2result(det_bboxes, det_labels, num_classes)
        return dict(bbox_results=bbox_result, track_results=track_result)
