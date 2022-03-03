# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmdet.models import LOSSES, weight_reduce_loss


@LOSSES.register_module()
class MultiPosCrossEntropyLoss(nn.Module):
    """multi-positive targets cross entropy loss."""

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(MultiPosCrossEntropyLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def multi_pos_cross_entropy(self,
                                pred,
                                label,
                                weight=None,
                                reduction='mean',
                                avg_factor=None):
        """
        Args:
            pred (torch.Tensor): The prediction.
            label (torch.Tensor): The assigned label of the prediction.
            weight (torch.Tensor): The element-wise weight.
            reduction (str): Same as built-in losses of PyTorch.
            avg_factor (float): Average factor when computing
                the mean of losses.
        Returns:
            torch.Tensor: Calculated loss
        """

        pos_inds = (label >= 1)
        neg_inds = (label == 0)
        pred_pos = pred * pos_inds.float()
        pred_neg = pred * neg_inds.float()
        # use -inf to mask out unwanted elements.
        pred_pos[neg_inds] = pred_pos[neg_inds] + float('inf')
        pred_neg[pos_inds] = pred_neg[pos_inds] + float('-inf')

        _pos_expand = torch.repeat_interleave(pred_pos, pred.shape[1], dim=1)
        _neg_expand = pred_neg.repeat(1, pred.shape[1])

        x = torch.nn.functional.pad((_neg_expand - _pos_expand), (0, 1),
                                    'constant', 0)
        loss = torch.logsumexp(x, dim=1)

        # apply weights and do the reduction
        if weight is not None:
            weight = weight.float()
        loss = weight_reduce_loss(
            loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

        return loss

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The classification score.
            label (torch.Tensor): The assigned label of the prediction.
            weight (torch.Tensor): The element-wise weight.
            avg_factor (float): Average factor when computing
                the mean of losses.
            reduction (str): Same as built-in losses of PyTorch.
        Returns:
            torch.Tensor: Calculated loss
        """
        assert cls_score.size() == label.size()
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_cls = self.loss_weight * self.multi_pos_cross_entropy(
            cls_score,
            label,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_cls
