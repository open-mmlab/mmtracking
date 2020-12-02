import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn.bricks import ConvModule
from mmdet.core.anchor import build_anchor_generator
from mmdet.models import HEADS

from mmtrack.core.correlation import depthwise_correlation


@HEADS.register_module()
class CorrelationHead(nn.Module):

    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 kernel_size=3,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 **kwargs):
        super(CorrelationHead, self).__init__()
        self.kernel_convs = ConvModule(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.search_convs = ConvModule(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.head_convs = nn.Sequential(
            ConvModule(
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_size=1,
                act_cfg=None))

    def forward(self, kernel, search):
        kernel = self.kernel_convs(kernel)
        search = self.search_convs(search)
        correlation_maps = depthwise_correlation(search, kernel)
        out = self.head_convs(correlation_maps)
        return out


@HEADS.register_module()
class MultiDepthwiseRPN(nn.Module):

    def __init__(self,
                 anchor_generator,
                 in_channels,
                 kernel_size=3,
                 norm_cfg=dict(type='BN'),
                 weighted_sum=False,
                 train_cfg=None,
                 test_cfg=None,
                 *args,
                 **kwargs):
        super(MultiDepthwiseRPN, self).__init__(*args, **kwargs)
        self.anchor_generator = build_anchor_generator(anchor_generator)
        self.anchors = self.anchor_generator.anchors
        self.window = self.anchor_generator.window
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.cls_heads = nn.ModuleList()
        self.reg_heads = nn.ModuleList()
        for i in range(len(in_channels)):
            self.cls_heads.append(
                CorrelationHead(in_channels[i], in_channels[i],
                                2 * self.anchor_generator.num_anchor,
                                kernel_size, norm_cfg))
            self.reg_heads.append(
                CorrelationHead(in_channels[i], in_channels[i],
                                4 * self.anchor_generator.num_anchor,
                                kernel_size, norm_cfg))

        self.weighted_sum = weighted_sum
        if self.weighted_sum:
            self.cls_weight = nn.Parameter(torch.ones(len(in_channels)))
            self.reg_weight = nn.Parameter(torch.ones(len(in_channels)))

    def forward(self, z_feats, x_feats):
        assert isinstance(z_feats, tuple) and isinstance(x_feats, tuple)
        assert len(z_feats) == len(x_feats) and len(z_feats) == len(
            self.cls_heads)

        if self.weighted_sum:
            cls_weight = nn.functional.softmax(self.cls_weight, dim=0)
            reg_weight = nn.functional.softmax(self.reg_weight, dim=0)
        else:
            reg_weight = cls_weight = [
                1.0 / len(z_feats) for i in range(len(z_feats))
            ]

        cls_score = 0
        bbox_pred = 0
        for i in range(len(z_feats)):
            cls_score_single = self.cls_heads[i](z_feats[i], x_feats[i])
            bbox_pred_single = self.reg_heads[i](z_feats[i], x_feats[i])
            cls_score += cls_weight[i] * cls_score_single
            bbox_pred += reg_weight[i] * bbox_pred_single

        return cls_score, bbox_pred

    def get_bbox(self, cls_score, bbox_pred, prev_bbox, scale_factor):
        if isinstance(self.anchors, np.ndarray):
            self.anchors = torch.from_numpy(self.anchors).to(cls_score.device)
        if isinstance(self.window, np.ndarray):
            self.window = torch.from_numpy(self.window).to(cls_score.device)

        cls_score = cls_score.permute(1, 2, 3, 0).contiguous()
        cls_score = cls_score.view(2, -1).permute(1, 0)
        cls_score = cls_score.softmax(dim=1)[:, 1]

        bbox_pred = bbox_pred.permute(1, 2, 3, 0).contiguous().view(4, -1)
        bbox_pred = bbox_pred.permute(1, 0)
        bbox_pred[:, 0] = bbox_pred[:, 0] * self.anchors[:, 2] + \
            self.anchors[:, 0]
        bbox_pred[:, 1] = bbox_pred[:, 1] * self.anchors[:, 3] + \
            self.anchors[:, 1]
        bbox_pred[:, 2] = torch.exp(bbox_pred[:, 2]) * self.anchors[:, 2]
        bbox_pred[:, 3] = torch.exp(bbox_pred[:, 3]) * self.anchors[:, 3]

        def change_ratio(ratio):
            return torch.max(ratio, 1. / ratio)

        def enlarge_size(w, h):
            pad = (w + h) * 0.5
            return torch.sqrt((w + pad) * (h + pad))

        # scale penalty
        scale_penalty = change_ratio(
            enlarge_size(bbox_pred[:, 2], bbox_pred[:, 3]) / enlarge_size(
                prev_bbox[2] * scale_factor, prev_bbox[3] * scale_factor))

        # aspect ratio penalty
        aspect_ratio_penalty = change_ratio(
            (prev_bbox[2] / prev_bbox[3]) /
            (bbox_pred[:, 2] / bbox_pred[:, 3]))

        # penalize cls_score
        penalty = torch.exp(-(aspect_ratio_penalty * scale_penalty - 1) *
                            self.test_cfg.penalty_k)
        penalty_score = penalty * cls_score

        # window penalty
        penalty_score = penalty_score * (1 - self.test_cfg.window_influence) \
            + self.window * self.test_cfg.window_influence

        best_idx = torch.argmax(penalty_score)
        best_score = cls_score[best_idx]
        best_bbox = bbox_pred[best_idx, :] / scale_factor

        # smooth bbox
        final_bbox = torch.zeros_like(best_bbox)
        lr = penalty[best_idx] * cls_score[best_idx] * self.test_cfg.lr
        final_bbox[0] = best_bbox[0] + prev_bbox[0]
        final_bbox[1] = best_bbox[1] + prev_bbox[1]
        final_bbox[2] = prev_bbox[2] * (1 - lr) + best_bbox[2] * lr
        final_bbox[3] = prev_bbox[3] * (1 - lr) + best_bbox[3] * lr

        return best_score, final_bbox
