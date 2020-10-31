from mmdet.core import bbox_overlaps

from ..builder import MODELS
from .qdtrack import QDTrack


@MODELS.register_module()
class TNT(QDTrack):

    def __init__(self,
                 auto_corr=False,
                 use_bbox_gt=False,
                 use_track_gt=False,
                 use_ref_rois=False,
                 use_ref_bbox_gt=False,
                 cross_nms_thr=0.5,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.auto_corr = auto_corr
        self.use_bbox_gt = use_bbox_gt
        self.use_track_gt = use_track_gt
        self.use_ref_rois = use_ref_rois
        self.use_ref_bbox_gt = use_ref_bbox_gt
        # for detected objects
        self.cross_nms_thr = cross_nms_thr

    def cross_nms(self, det_bboxes, det_labels):
        out_bboxes = []
        out_labels = []
        for bboxes, labels in zip(det_bboxes, det_labels):
            valids = bboxes.new_ones((bboxes.size(0)))
            ious = bbox_overlaps(bboxes[:, :-1], bboxes[:, :-1])
            for i in range(1, bboxes.size(0)):
                if (ious[i, :i] > self.cross_nms_thr).any():
                    valids[i] = 0
                    ious[:, i] = 0
            bboxes = bboxes[valids]
            labels = labels[valids]
            out_bboxes.append(bboxes)
            out_labels.append(labels)
        return out_bboxes, out_labels

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
        num_imgs = len(img_metas)
        # feature extraction
        x = self.detector.extract_feat(img)
        ref_x = self.detector.extract_feat(ref_img)

        if self.use_track_gt:
            for i in range(num_imgs):
                valid_inds = gt_match_indices[i] > -1
                gt_bboxes[i] = gt_bboxes[i][valid_inds]
                gt_labels[i] = gt_labels[i][valid_inds]

        if self.auto_corr or (not self.use_bbox_gt):
            key_proposals = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            key_proposals = gt_bboxes.copy()

        if self.use_ref_rois or (not self.use_ref_bbox_gt):
            ref_proposals = self.rpn_head.simple_test_rpn(ref_x, ref_img_metas)
        else:
            ref_proposals = ref_gt_bboxes.copy()

        if not self.use_bbox_gt:
            key_bboxes, key_labels = self.roi_head.simple_test_detector(
                x, img_metas, key_proposals, rescale=False)
            gt_bboxes, gt_labels = self.cross_nms(key_bboxes, key_labels)

        if not self.use_ref_bbox_gt:
            ref_bboxes, ref_labels = self.roi_head.simple_test_detector(
                ref_x, ref_img_metas, key_proposals, rescale=False)
            ref_gt_bboxes, ref_gt_labels = self.cross_nms(
                ref_bboxes, ref_labels)

        loss_track = self.track_head.forward_train(
            x, img_metas, key_proposals, gt_bboxes, gt_labels, ref_x,
            ref_img_metas, ref_proposals, ref_gt_bboxes, ref_gt_labels,
            gt_bboxes_ignore, ref_gt_bboxes_ignore)

        losses.update(loss_track)
        return losses
