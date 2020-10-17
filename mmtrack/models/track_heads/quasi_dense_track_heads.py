import torch
import torch.nn as nn
from mmdet.core import bbox2roi, build_assigner, build_sampler
from mmdet.models import HEADS, build_head, build_roi_extractor


@HEADS.register_module()
class QuasiDenseTrackHead(nn.Module):

    def __init__(self,
                 multi_positive=True,
                 roi_extractor=None,
                 roi_assigner=None,
                 roi_sampler=None,
                 embed_head=None):
        super().__init__()
        self.multi_positive = multi_positive
        self.roi_extractor = build_roi_extractor(roi_extractor)
        self.roi_assigner = build_assigner(self.track_train_cfg.assigner)
        self.roi_sampler = build_sampler(
            self.track_train_cfg.sampler, context=self)
        self.embed_head = build_head(embed_head)

    @property
    def with_roi_extractor(self):
        """bool: whether the RoI head contains a `embed_head`"""
        return hasattr(self,
                       'roi_extractor') and self.roi_extractor is not None

    @property
    def with_roi_assigner(self):
        """bool: whether the RoI head contains a `embed_head`"""
        return hasattr(self, 'roi_assigner') and self.roi_assigner is not None

    @property
    def with_roi_sampler(self):
        """bool: whether the RoI head contains a `embed_head`"""
        return hasattr(self, 'roi_sampler') and self.roi_sampler is not None

    @property
    def with_embed(self):
        """bool: whether the RoI head contains a `embed_head`"""
        return hasattr(self, 'embed_head') and self.embed_head is not None

    def init_weights(self):
        if self.with_roi_extractor:
            self.roi_extractor.init_weights()
        if self.with_embed:
            self.embed_head.init_weights()

    def forward_train(self,
                      x,
                      img_metas,
                      proposals,
                      gt_bboxes,
                      gt_labels,
                      gt_match_indices,
                      ref_x,
                      ref_img_metas,
                      ref_proposals,
                      ref_gt_bboxes,
                      ref_gt_labels,
                      ref_gt_match_indices,
                      gt_bboxes_ignore=None,
                      ref_gt_bboxes_ignore=None):
        num_imgs = len(img_metas)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        if ref_gt_bboxes_ignore is None:
            ref_gt_bboxes_ignore = [None for _ in range(num_imgs)]
        key_sampling_results, ref_sampling_results = [], []
        for i in range(num_imgs):
            assign_result = self.roi_assigner.assign(proposals[i],
                                                     gt_bboxes[i],
                                                     gt_bboxes_ignore[i],
                                                     gt_labels[i])
            sampling_result = self.roi_sampler.sample(
                assign_result,
                proposals[i],
                gt_bboxes[i],
                gt_labels[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            key_sampling_results.append(sampling_result)

            ref_assign_result = self.roi_assigner.assign(
                ref_proposals[i], ref_gt_bboxes[i], ref_gt_bboxes_ignore[i],
                ref_gt_labels[i])
            ref_sampling_result = self.roi_sampler.sample(
                ref_assign_result,
                ref_proposals[i],
                ref_gt_bboxes[i],
                ref_gt_labels[i],
                feats=[lvl_feat[i][None] for lvl_feat in ref_x])
            ref_sampling_results.append(ref_sampling_result)

        # positive samples on key frame
        key_bboxes = [res.pos_bboxes for res in key_sampling_results]
        key_feats = self._track_forward(x, key_bboxes, split=True)

        # samples on reference frames
        if self.multi_positive:
            ref_bboxes = [
                torch.cat((gts, res.bboxes), dim=0)
                for gts, res in zip(ref_gt_bboxes, ref_sampling_results)
            ]
        else:
            ref_bboxes = [
                torch.cat((gts, res.neg_bboxes), dim=0)
                for gts, res in zip(ref_gt_bboxes, ref_sampling_results)
            ]
        ref_feats = self._track_forward(ref_x, ref_bboxes, split=True)

        match_feats = self.track_head.match(key_feats, ref_feats)
        asso_targets = self.track_head.get_track_targets(
            gt_match_indices, ref_gt_bboxes, key_sampling_results,
            ref_sampling_results, self.multi_positive)
        loss_track = self.track_head.loss(*match_feats, *asso_targets)
        return loss_track

    def _track_forward(self, x, bboxes, split=False):
        """Track head forward function used in both training and testing."""
        rois = bbox2roi(bboxes)
        track_feats = self.track_roi_extractor(
            x[:self.track_roi_extractor.num_inputs], rois)
        track_feats = self.track_head(track_feats)
        if split:
            nums = [b.size(0) for b in bboxes]
            track_feats = torch.split(track_feats, nums)
        return track_feats

    def simple_test(self, x, img_metas, det_bboxes, rescale):
        if det_bboxes.size(0) == 0:
            return None
        track_bboxes = det_bboxes[:, :-1] * torch.tensor(
            img_metas[0]['scale_factor']).to(det_bboxes.device)
        track_feats = self._track_forward(x, [track_bboxes])
        return track_feats
