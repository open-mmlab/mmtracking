# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.core import bbox2roi
from mmdet.models import HEADS

from .roi_track_head import RoITrackHead


@HEADS.register_module()
class QuasiDenseTrackHead(RoITrackHead):
    """The quasi-dense track head."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_match_indices,
                      ref_x,
                      ref_img_metas,
                      ref_proposals,
                      ref_gt_bboxes,
                      ref_gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      ref_gt_bboxes_ignore=None,
                      ref_gt_mask=None,
                      *args,
                      **kwargs):
        """Forward function during training.

         Args:
            x (list[Tensor]): list of multi-level image features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            proposal_list (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes of the image,
                each item has a shape (num_gts, 4).
            gt_labels (list[Tensor]): Ground truth labels of all images.
                each has a shape (num_gts,).
            gt_match_indices (list(Tensor)): Mapping from gt_instance_ids to
                ref_gt_instance_ids of the same tracklet in a pair of images.
            ref_x (list[Tensor]): list of multi-level ref_img features.
            ref_img_metas (list[dict]): list of reference image info dict where
                each dict has: 'img_shape', 'scale_factor', 'flip', and may
                also contain 'filename', 'ori_shape', 'pad_shape',
                and 'img_norm_cfg'.
            ref_proposal_list (list[Tensors]): list of ref_img
                region proposals.
            ref_gt_bboxes (list[Tensor]): Ground truth bboxes of the
                reference image, each item has a shape (num_gts, 4).
            ref_gt_labels (list[Tensor]): Ground truth labels of all
                reference images, each has a shape (num_gts,).
            gt_bboxes_ignore (list[Tensor], None): Ground truth bboxes to be
                ignored, each item has a shape (num_ignored_gts, 4).
            gt_masks (list[Tensor]) : Masks for each bbox, has a shape
                (num_gts, h , w).
            ref_gt_bboxes_ignore (list[Tensor], None): Ground truth bboxes
                of reference images to be ignored,
                each item has a shape (num_ignored_gts, 4).
            ref_gt_masks (list[Tensor]) : Masks for each reference bbox,
                has a shape (num_gts, h , w).

        Returns:
            dict[str : Tensor]: Track losses.
        """

        assert self.with_track
        num_imgs = len(img_metas)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        if ref_gt_bboxes_ignore is None:
            ref_gt_bboxes_ignore = [None for _ in range(num_imgs)]
        key_sampling_results, ref_sampling_results = [], []
        for i in range(num_imgs):
            assign_result = self.bbox_assigner.assign(proposal_list[i],
                                                      gt_bboxes[i],
                                                      gt_bboxes_ignore[i],
                                                      gt_labels[i])
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                proposal_list[i],
                gt_bboxes[i],
                gt_labels[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            key_sampling_results.append(sampling_result)

            ref_assign_result = self.bbox_assigner.assign(
                ref_proposals[i], ref_gt_bboxes[i], ref_gt_bboxes_ignore[i],
                ref_gt_labels[i])
            ref_sampling_result = self.bbox_sampler.sample(
                ref_assign_result,
                ref_proposals[i],
                ref_gt_bboxes[i],
                ref_gt_labels[i],
                feats=[lvl_feat[i][None] for lvl_feat in ref_x])
            ref_sampling_results.append(ref_sampling_result)

        key_bboxes = [res.pos_bboxes for res in key_sampling_results]
        key_feats = self.extract_bbox_feats(x, key_bboxes)
        ref_bboxes = [res.bboxes for res in ref_sampling_results]
        ref_feats = self.extract_bbox_feats(ref_x, ref_bboxes)

        match_feats = self.embed_head.match(key_feats, ref_feats,
                                            key_sampling_results,
                                            ref_sampling_results)
        asso_targets = self.embed_head.get_targets(gt_match_indices,
                                                   key_sampling_results,
                                                   ref_sampling_results)
        loss_track = self.embed_head.loss(*match_feats, *asso_targets)

        return loss_track

    def extract_bbox_feats(self, x, bboxes):
        """Extract roi features."""

        rois = bbox2roi(bboxes)
        track_feats = self.roi_extractor(x[:self.roi_extractor.num_inputs],
                                         rois)
        track_feats = self.embed_head(track_feats)
        return track_feats
