import torch
from mmdet.models import HEADS

from .quasi_dense_track_head import QuasiDenseTrackHead


def sample_gumbel(prob):
    unif = torch.distributions.Uniform(0, 1).sample(prob.shape)
    gumbel = -torch.log(-torch.log(unif)).to(prob.device)
    return gumbel


def gumbel_softmax(prob, temperature=0.1):
    gumbel = sample_gumbel(prob)
    prob = (gumbel + torch.log(prob)) / temperature
    prob = prob.softmax(dim=1)
    return prob


@HEADS.register_module()
class TrackNoTrackHead(QuasiDenseTrackHead):

    def __init__(self,
                 match_method='dense_aggregation',
                 match_with_gumbel=False,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        # matching
        self.match_method = match_method
        assert match_method in [
            'sparse_aggregation', 'dense_aggregation', 'cycle_dot_product',
            'dual_softmax'
        ]
        self.match_with_gumbel = match_with_gumbel

    def forward_train(self,
                      x,
                      img_metas,
                      proposals,
                      gt_bboxes,
                      gt_labels,
                      ref_x,
                      ref_img_metas,
                      ref_proposals,
                      ref_gt_bboxes,
                      ref_gt_labels,
                      gt_bboxes_ignore=None,
                      ref_gt_bboxes_ignore=None):
        num_imgs = len(img_metas)

        gt_bboxes_ignore = [None for _ in range(num_imgs)]
        ref_gt_bboxes_ignore = [None for _ in range(num_imgs)]
        key_sampling_results, ref_sampling_results = [], []
        for i in range(num_imgs):
            assign_result = self.roi_assigner.assign(proposals[i],
                                                     gt_bboxes[i, :4],
                                                     gt_bboxes_ignore[i],
                                                     gt_labels[i])
            sampling_result = self.roi_sampler.sample(
                assign_result,
                proposals[i],
                gt_bboxes[i, :4],
                gt_labels[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            key_sampling_results.append(sampling_result)

            ref_assign_result = self.track_roi_assigner.assign(
                ref_proposals[i], ref_gt_bboxes[i, :4],
                ref_gt_bboxes_ignore[i], ref_gt_labels[i])
            ref_sampling_result = self.track_roi_sampler.sample(
                ref_assign_result,
                ref_proposals[i],
                ref_gt_bboxes[i, :4],
                ref_gt_labels[i],
                feats=[lvl_feat[i][None] for lvl_feat in ref_x])
            ref_sampling_results.append(ref_sampling_result)

        # loss targets
        self_match_indices = [
            torch.arange(_.size(0)).to(_.device) for _ in gt_bboxes
        ]
        asso_targets = self.embed_head.get_track_targets(
            self_match_indices, gt_bboxes, key_sampling_results,
            key_sampling_results, self.multi_positive)

        # positive samples on key frame
        key_pos_bboxes = [res.pos_bboxes for res in key_sampling_results]
        key_pos_embeds = self._track_forward(x, key_pos_bboxes, split=True)
        # all samples on key frame
        if self.multi_positive:
            ref_bboxes = [
                torch.cat((gts, res.bboxes), dim=0)
                for gts, res in zip(gt_bboxes, key_sampling_results)
            ]
        else:
            ref_bboxes = [
                torch.cat((gts, res.neg_bboxes), dim=0)
                for gts, res in zip(gt_bboxes, key_sampling_results)
            ]
        key_dense_embeds = self._track_forward(ref_x, ref_bboxes, split=True)

        matcher = getattr(self, self.match_method)
        match_feats = matcher(x, key_pos_embeds, key_dense_embeds, ref_x,
                              ref_sampling_results, gt_bboxes, ref_gt_bboxes)

        loss_track = self.embed_head.loss(*match_feats, *asso_targets)
        return loss_track

    def dual_softmax(
            self,
            key_pos_embeds=None,  # P0
            key_sparse_embeds=None,  # S0
            key_dense_embeds=None,  # D0
            ref_pos_embeds=None,  # P1
            ref_sparse_embeds=None,  # S1
            ref_dense_embeds=None):  # D1
        pass

    def cycle_dot_product(
            self,
            key_pos_embeds=None,  # P0
            key_sparse_embeds=None,  # S0
            key_dense_embeds=None,  # D0
            ref_pos_embeds=None,  # P1
            ref_sparse_embeds=None,  # S1
            ref_dense_embeds=None):  # D1
        # return [P0, N0]

        # forward
        kp2rs = torch.mm(key_pos_embeds, ref_sparse_embeds.t())  # [P0, S1]
        kp2rs = kp2rs.softmax(dim=1)
        rs2rp = torch.mm(ref_sparse_embeds, ref_pos_embeds.t())  # [S1, P1]
        kp2rp = torch.mm(kp2rs, rs2rp)  # [P0, P1]

        # backward
        rp2ks = torch.mm(ref_pos_embeds, key_sparse_embeds.t())  # [P1, S0]
        rp2ks = rp2ks.softmax(dim=1)
        ks2kd = torch.mm(key_sparse_embeds, key_dense_embeds.t())  # [S0, D0]
        rp2kd = torch.mm(rp2ks, ks2kd)  # [P1, D0]

        kp2kd = torch.mm(kp2rp, rp2kd)

        return kp2kd

    def dense_aggregation(self, key_pos_embeds, key_dense_embeds, ref_x,
                          ref_sampling_results, gt_bboxes, ref_gt_bboxes):
        ref_bboxes = [res.bboxes for res in ref_sampling_results]
        ref_dense_embeds = self._track_forward(ref_x, ref_bboxes, split=True)
        pos2dense = torch.mm(key_pos_embeds, ref_dense_embeds.t())
        pos2dense = pos2dense.softmax(dim=1)
        agg_k = torch.mm(pos2dense, ref_dense_embeds)
        match_feats = self.embed_head.match(agg_k, key_dense_embeds)
        return match_feats

    def sparse_aggregation(self, key_pos_embeds, key_dense_embeds, ref_x,
                           ref_sampling_results, gt_bboxes, ref_gt_bboxes):
        ref_sparse_embeds = self._track_forward(
            ref_x, ref_gt_bboxes, split=True)
        pos2sparse = torch.mm(key_pos_embeds, ref_sparse_embeds.t())
        pos2sparse = pos2sparse.softmax(dim=1)
        if self.match_with_gumbel:
            pos2sparse = gumbel_softmax(pos2sparse)
        agg_k = torch.mm(pos2sparse, ref_sparse_embeds)
        match_feats = self.embed_head.match(agg_k, key_dense_embeds)
        return match_feats
