import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmdet.models import HEADS, build_loss

from mmtrack.core import embed_similarity


@HEADS.register_module()
class QuasiDenseEmbedHead(nn.Module):

    def __init__(self,
                 num_convs=4,
                 num_fcs=1,
                 roi_feat_size=7,
                 in_channels=256,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 embed_channels=256,
                 conv_cfg=None,
                 norm_cfg=None,
                 softmax_temperature=-1,
                 loss_track=dict(
                     type='MultiPosCrossEntropyLoss', loss_weight=0.25),
                 loss_track_aux=dict(
                     type='L2Loss',
                     neg_pos_ub=3,
                     pos_margin=0,
                     neg_margin=0.3,
                     hard_mining=True,
                     loss_weight=1.0),
                 num_ids=None,
                 loss_id=None):
        super(QuasiDenseEmbedHead, self).__init__()
        self.num_convs = num_convs
        self.num_fcs = num_fcs
        self.roi_feat_size = roi_feat_size
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.embed_channels = embed_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.relu = nn.ReLU(inplace=True)
        self.convs, self.fcs, last_layer_dim = self._add_conv_fc_branch(
            self.num_convs, self.num_fcs, self.in_channels)
        self.fc_embed = nn.Linear(last_layer_dim, embed_channels)

        self.softmax_temperature = softmax_temperature
        if loss_track is not None:
            self.loss_track = build_loss(loss_track)
        else:
            self.loss_track = None
        if loss_track_aux is not None:
            self.loss_track_aux = build_loss(loss_track_aux)
        else:
            self.loss_track_aux = None
        if loss_id is not None:
            assert num_ids > 0
            self.loss_id = build_loss(loss_id)
            self.fc_id = nn.Linear(embed_channels, num_ids, bias=False)
        else:
            self.loss_id = None

    def _add_conv_fc_branch(self, num_convs, num_fcs, in_channels):
        last_layer_dim = in_channels
        # add branch specific conv layers
        convs = nn.ModuleList()
        if num_convs > 0:
            for i in range(num_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        fcs = nn.ModuleList()
        if num_fcs > 0:
            last_layer_dim *= (self.roi_feat_size * self.roi_feat_size)
            for i in range(num_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                fcs.append(nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return convs, fcs, last_layer_dim

    def init_weights(self):
        for m in self.fcs:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.fc_embed.weight, 0, 0.01)
        nn.init.constant_(self.fc_embed.bias, 0)
        if hasattr(self, 'fc_id'):
            nn.init.kaiming_normal_(self.fc_id.weight)

    def forward(self, x):
        if x.numel() == 0:
            return x
        if self.num_convs > 0:
            for i, conv in enumerate(self.convs):
                x = conv(x)
        x = x.view(x.size(0), -1)
        if self.num_fcs > 0:
            for i, fc in enumerate(self.fcs):
                x = self.relu(fc(x))
        x = self.fc_embed(x)
        return x

    def get_track_targets(self,
                          gt_match_indices,
                          ref_gt_bboxes,
                          key_sampling_results,
                          ref_sampling_results,
                          multi_positive=True):
        track_targets = []
        track_weights = []
        for indices, ref_gts, key_res, ref_res in zip(gt_match_indices,
                                                      ref_gt_bboxes,
                                                      key_sampling_results,
                                                      ref_sampling_results):
            ref_gt_inds = torch.arange(ref_gts.size(0)).to(indices.device)
            if multi_positive:
                num_refs = ref_gts.size(0) + ref_res.bboxes.size(0)
                ref_gt_inds = torch.cat(
                    (ref_gt_inds, ref_res.pos_assigned_gt_inds))
            else:
                num_refs = ref_gts.size(0) + ref_res.neg_bboxes.size(0)
            num_keys = key_res.pos_bboxes.size(0)
            targets = indices.new_zeros((num_keys, num_refs), dtype=torch.int)
            match_indices = indices[key_res.pos_assigned_gt_inds]
            pos2pos = match_indices.view(-1, 1) == ref_gt_inds.view(1, -1)
            targets[:, :pos2pos.size(1)] = pos2pos.int()
            weights = (targets.sum(dim=1) > 0).float()
            track_targets.append(targets)
            track_weights.append(weights)
        return track_targets, track_weights

    def match(self, key_embeds, ref_embeds):
        sims, cos_sims = [], []
        for key_embed, ref_embed in zip(key_embeds, ref_embeds):
            sim = embed_similarity(
                key_embed,
                ref_embed,
                method='dot_product',
                temperature=self.softmax_temperature)
            sims.append(sim)
            if self.loss_track_aux is not None:
                cos_sim = embed_similarity(
                    key_embed, ref_embed, method='cosine')
                cos_sims.append(cos_sim)
            else:
                cos_sims.append(None)
        return sims, cos_sims

    def loss(self, sims, cos_sims, targets, weights):
        losses = dict()
        loss_track = torch.tensor([0.], device=sims[0].device)
        loss_track_aux = torch.tensor([0.], device=sims[0].device)
        nums = len(sims)
        for _sims, _cos_sims, _targets, _weights in zip(
                sims, cos_sims, targets, weights):
            if _targets.numel() > 0:
                if self.loss_track is not None:
                    loss_track += self.loss_track(
                        _sims, _targets, _weights, avg_factor=_weights.sum())
                if self.loss_track_aux is not None:
                    loss_track_aux += self.loss_track_aux(_cos_sims, _targets)
        if self.loss_track is not None:
            losses['loss_track'] = loss_track / nums
        if self.loss_track_aux is not None:
            losses['loss_track_aux'] = loss_track_aux / nums
        return losses

    def cal_loss_id(self, key_feats, gt_instance_ids, key_sampling_results):
        key_feats = torch.cat(key_feats, dim=0)
        logits = embed_similarity(
            key_feats,
            self.fc_id.weight,
            method='dot_product',
            temperature=self.softmax_temperature)
        pos_gt_inds = [_.pos_assigned_gt_inds for _ in key_sampling_results]
        gt_ids = [
            _ids[_inds] for (_ids, _inds) in zip(gt_instance_ids, pos_gt_inds)
        ]
        gt_ids = torch.cat(gt_ids, dim=0)
        weights = torch.ones_like(gt_ids)
        weights[gt_ids < 0] = 0
        loss_id = self.loss_id(logits, gt_ids, weight=weights)
        return dict(loss_id=loss_id)

    @staticmethod
    def random_choice(gallery, num):
        """Random select some elements from the gallery.

        It seems that Pytorch's implementation is slower than numpy so we use
        numpy to randperm the indices.
        """
        assert len(gallery) >= num
        if isinstance(gallery, list):
            gallery = np.array(gallery)
        cands = np.arange(len(gallery))
        np.random.shuffle(cands)
        rand_inds = cands[:num]
        if not isinstance(gallery, np.ndarray):
            rand_inds = torch.from_numpy(rand_inds).long().to(gallery.device)
        return gallery[rand_inds]
