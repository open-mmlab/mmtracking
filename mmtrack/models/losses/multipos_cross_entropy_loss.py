import torch
import torch.nn as nn
from mmdet.models import LOSSES, weight_reduce_loss


def multi_pos_cross_entropy(pred,
                            label,
                            weight=None,
                            reduction='mean',
                            avg_factor=None):
    # element-wise losses
    # pos_inds = (label == 1).float()
    # neg_inds = (label == 0).float()
    # exp_pos = (torch.exp(-1 * pred) * pos_inds).sum(dim=1)
    # exp_neg = (torch.exp(pred.clamp(max=80)) * neg_inds).sum(dim=1)
    # loss = torch.log(1 + exp_pos * exp_neg)

    # a more numerical stable implementation.
    pos_inds = (label == 1)
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


@LOSSES.register_module()
class MultiPosCrossEntropyLoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(MultiPosCrossEntropyLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert cls_score.size() == label.size()
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_cls = self.loss_weight * multi_pos_cross_entropy(
            cls_score,
            label,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_cls
