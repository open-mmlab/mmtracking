# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.ops.prroi_pool import PrRoIPool
from mmcv.runner.base_module import BaseModule
from mmdet.models import HEADS
from mmdet.models.builder import build_head, build_loss

from mmtrack.models.track_heads.stark_head import ScoreHead as MLPScoreHead


@HEADS.register_module()
class MixFormerScoreDecoder(nn.Module):
    """Score Prediction Module (SPM) proposed in
    "MixFormer: End-to-End Tracking with Iterative
    Mixed Attention". `MixFormer <https://arxiv.org/abs/2203.11082>`_.

    Args:
        pool_size (int): pool size for roi pooling
        feat_size (int): search region feature map size
        stride (int): ratio between original image size
        and feature map size
        num_heads (int): number of heads of attention
        hidden_dim (int): embedding dimension
        num_layer (int): number of layers of the mlp
    """

    def __init__(self,
                 pool_size=4,
                 feat_size=20,
                 stride=16,
                 num_heads=6,
                 hidden_dim=384,
                 num_layers=3):
        super().__init__()
        self.feat_size = feat_size
        self.stride = stride
        self.img_sz = feat_size * stride
        self.num_heads = num_heads
        self.pool_size = pool_size
        self.score_head = MLPScoreHead(hidden_dim, hidden_dim, 1, num_layers)
        self.scale = hidden_dim**-0.5
        self.search_prroipool = PrRoIPool(pool_size, spatial_scale=1.0)
        self.proj_q = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim, bias=True) for _ in range(2))
        self.proj_k = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim, bias=True) for _ in range(2))
        self.proj_v = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim, bias=True) for _ in range(2))

        self.proj = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim, bias=True) for _ in range(2))

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(2))

        self.score_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        trunc_normal_(self.score_token, std=.02)

    def forward(self, search_feat, template_feat, search_box):
        """
        Args:
            search_feat (Tensor): Search region features extracted from
            backbone with shape (N, C, H, W).
            template_feat (Tensor): Template features extracted from
            backbone with shape (N, C, H, W).
            search_box (Tensor): of shape (B, 4), in
            [tl_x, tl_y, br_x, br_y] format.
        Returns:
            out_score (Tensor): Confidence score of the predicted result.
                of shape (b, 1, 1)
        """
        b, c, h, w = search_feat.shape
        search_box = search_box.clone() / self.img_sz * w
        # bb_pool = box_cxcywh_to_xyxy(search_box.view(-1, 4))
        bb_pool = search_box.view(-1, 4)
        # Add batch_index to rois
        batch_size = bb_pool.shape[0]
        batch_index = torch.arange(
            batch_size, dtype=torch.float32).view(-1, 1).to(bb_pool.device)
        target_roi = torch.cat((batch_index, bb_pool), dim=1)

        # decoder1: query for search_box feat
        # decoder2: query for template feat
        x = self.score_token.expand(b, -1, -1)
        x = self.norm1(x)
        search_box_feat = rearrange(
            self.search_prroipool(search_feat, target_roi),
            'b c h w -> b (h w) c')
        template_feat = rearrange(template_feat, 'b c h w -> b (h w) c')
        kv_memory = [search_box_feat, template_feat]
        for i in range(len(kv_memory)):
            q = rearrange(
                self.proj_q[i](x), 'b t (n d) -> b n t d', n=self.num_heads)
            k = rearrange(
                self.proj_k[i](kv_memory[i]),
                'b t (n d) -> b n t d',
                n=self.num_heads)
            v = rearrange(
                self.proj_v[i](kv_memory[i]),
                'b t (n d) -> b n t d',
                n=self.num_heads)

            attn_score = torch.einsum('bhlk,bhtk->bhlt', [q, k]) * self.scale
            attn = F.softmax(attn_score, dim=-1)
            x = torch.einsum('bhlt,bhtv->bhlv', [attn, v])
            x = rearrange(x, 'b h t d -> b t (h d)')  # (b, 1, c)
            x = self.proj[i](x)
            x = self.norm2[i](x)
        out_scores = self.score_head(x)  # (b, 1, 1)

        return out_scores


@HEADS.register_module()
class MixFormerHead(BaseModule):
    """MixFormer head module for bounding box regression and prediction of
    confidence of tracking bbox.

    This module is proposed in "MixFormer: End-to-End Tracking with Iterative
    Mixed Attention". `MixFormer <https://arxiv.org/abs/2203.11082>`_.
    """

    def __init__(self,
                 bbox_head=None,
                 score_head=None,
                 loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                 loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(MixFormerHead, self).__init__(init_cfg=init_cfg)

        assert bbox_head is not None
        self.bbox_head = build_head(bbox_head)
        self.score_decoder_head = build_head(score_head)

        self.loss_iou = build_loss(loss_iou)
        self.loss_bbox = build_loss(loss_bbox)

    def forward_bbox_head(self, search):
        """
        Args:
            search (Tensor): Search region features extracted from backbone,
            with shape (N, C, H, W).
        Returns:
            Tensor: of shape (bs, 1, 4). The bbox is in
            [tl_x, tl_y, br_x, by_y] format.
        """
        b = search.shape[0]
        outputs_coord = self.bbox_head(search)
        outputs_coord = outputs_coord.view(b, 1, 4)
        return outputs_coord

    def forward(self, template, search, run_score_head=True, gt_bboxes=None):
        """
        Args:
            template (Tensor): Template features extracted from backbone,
            with shape (N, C, H, W).
            search (Tensor): Search region features extracted from backbone,
            with shape (N, C, H, W).
        Returns:
            (dict):
                - 'pred_bboxes': (Tensor) of shape (bs, 1, 4), in
                    [tl_x, tl_y, br_x, br_y] format
                - 'pred_scores': (Tensor) of shape (bs, 1, 1)
        """

        track_results = {}
        outputs_coord = self.forward_bbox_head(search)
        track_results['pred_bboxes'] = outputs_coord

        if run_score_head:
            if gt_bboxes is None:
                gt_bboxes = outputs_coord.clone().view(-1, 4)
            pred_scores = self.score_decoder_head(search, template, gt_bboxes)
            track_results['pred_scores'] = pred_scores

        return track_results

    def loss(self, track_results, gt_bboxes, gt_labels, img_size=None):
        """compute loss. Not Implemented yet!

        Args:
            track_results (dict): it may contains the following keys:
                - 'pred_bboxes': bboxes of (N, num_query, 4) shape in
                    [tl_x, tl_y, br_x, br_y] format.
                - 'pred_scores': scores of (N, num_query, 1) shaoe.
            gt_bboxes (list[Tensor]): ground truth bboxes for search image
                with shape (N, 5) in [0., tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): ground truth labels for
                search imges with shape (N, 2).
            img_size (tuple, optional): the size (h, w) of original
                search image. Defaults to None.
        """
        raise NotImplementedError
        pred_bboxes = track_results['pred_bboxes']
        if torch.isnan(pred_bboxes).any():
            raise ValueError('Network outputs is Nan! Stop training')
        pred_bboxes = pred_bboxes.view(-1, 4)
        gt_bboxes = torch.cat(
            gt_bboxes, dim=0).type(torch.float32)[:, 1:]  # (N, 4)
        gt_bboxes[:, 0:4:2] = gt_bboxes[:, 0:4:2] / float(img_size[1])
        gt_bboxes[:, 1:4:2] = gt_bboxes[:, 1:4:2] / float(img_size[0])
        gt_bboxes = gt_bboxes.clamp(0., 1.)

        # compute giou loss
        try:
            giou_loss, iou = self.loss_iou(pred_bboxes,
                                           gt_bboxes)  # (BN,4) (BN,4)
        except Exception:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.loss_bbox(pred_bboxes, gt_bboxes)

        if 'pred_scores' in track_results:
            raise NotImplementedError
        else:
            status = {'Loss/giou': giou_loss, 'iou': iou, 'Loss/l1': l1_loss}
            return status
