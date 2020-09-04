import torch
from mmdet.core import build_assigner, build_sampler, bbox2roi
from mmdet.models import HEADS, build_head, build_roi_extractor
from mmdet.models.roi_heads import StandardRoIHead


@HEADS.register_module()
class QuasiDenseRoIHead(StandardRoIHead):

    def __init__(self,
                 track_roi_extractor=None,
                 track_head=None,
                 track_train_cfg=None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        assert track_head is not None
        self.track_train_cfg = track_train_cfg
        self.init_track_head(track_roi_extractor, track_head)
        self.init_track_assigner_sampler()

    def init_track_assigner_sampler(self):
        """Initialize assigner and sampler."""
        if self.track_train_cfg.get('assigner', None):
            self.track_roi_assigner = build_assigner(
                self.track_train_cfg.assigner)
            self.track_share_assigner = False
        else:
            self.track_roi_assigner = self.bbox_assigner
            self.track_share_assigner = True

        if self.track_train_cfg.get('sampler', None):
            self.track_roi_sampler = build_sampler(
                self.track_train_cfg.sampler, context=self)
            self.track_share_sampler = False
        else:
            self.track_roi_sampler = self.bbox_sampler
            self.track_share_sampler = True

    @property
    def with_track(self):
        """bool: whether the RoI head contains a `track_head`"""
        return hasattr(self, 'track_head') and self.track_head is not None

    def init_track_head(self, track_roi_extractor, track_head):
        """Initialize ``track_head``"""
        if track_roi_extractor is not None:
            self.track_roi_extractor = build_roi_extractor(track_roi_extractor)
            self.track_share_extractor = False
        else:
            self.track_share_extractor = True
            self.track_roi_extractor = self.bbox_roi_extractor
        self.track_head = build_head(track_head)

    def init_weights(self, *args, **kwargs):
        super().init_weights(*args, **kwargs)
        if self.with_track:
            self.track_head.init_weights()
            if not self.track_share_extractor:
                self.track_roi_extractor.init_weights()

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_mids,
                      ref_x,
                      ref_img_metas,
                      ref_proposals,
                      ref_gt_bboxes,
                      ref_gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      ref_gt_bboxes_ignore=None,
                      *args,
                      **kwargs):
        losses = super().forward_train(x, img_metas, proposal_list, gt_bboxes,
                                       gt_labels, gt_bboxes_ignore, gt_masks,
                                       *args, **kwargs)

        num_imgs = len(img_metas)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        if ref_gt_bboxes_ignore is None:
            ref_gt_bboxes_ignore = [None for _ in range(num_imgs)]
        key_sampling_results, ref_sampling_results = [], []
        for i in range(num_imgs):
            assign_result = self.track_roi_assigner.assign(
                proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                gt_labels[i])
            sampling_result = self.track_roi_sampler.sample(
                assign_result,
                proposal_list[i],
                gt_bboxes[i],
                gt_labels[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            key_sampling_results.append(sampling_result)

            ref_assign_result = self.track_roi_assigner.assign(
                ref_proposals[i], ref_gt_bboxes[i], ref_gt_bboxes_ignore[i],
                ref_gt_labels[i])
            ref_sampling_result = self.track_roi_sampler.sample(
                ref_assign_result,
                ref_proposals[i],
                ref_gt_bboxes[i],
                ref_gt_labels[i],
                feats=[lvl_feat[i][None] for lvl_feat in ref_x])
            ref_sampling_results.append(ref_sampling_result)

        key_pos_rois = bbox2roi(
            [res.pos_bboxes for res in key_sampling_results])
        key_pos_feats = self.track_roi_extractor(
            x[:self.track_roi_extractor.num_inputs], key_pos_rois)
        key_feats = self.track_head(key_pos_feats)

        ref_rois = bbox2roi([res.bboxes for res in ref_sampling_results])
        ref_feats = self.track_roi_extractor(
            ref_x[:self.track_roi_extractor.num_inputs], ref_rois)
        ref_feats = self.track_head(ref_feats)

        match_feats = self.track_head.match(key_feats, ref_feats,
                                            key_sampling_results,
                                            ref_sampling_results)
        asso_targets = self.track_head.get_track_targets(
            gt_mids, key_sampling_results, ref_sampling_results)
        loss_track = self.track_head.loss(*match_feats, *asso_targets)
        
        losses.update(loss_track)

        return losses
