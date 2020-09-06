import numpy as np
import torch
import torch.nn as nn
from mmdet.models import LOSSES, weighted_loss


@weighted_loss
def l2_loss(pred, target):
    """L2 loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.

    Returns:
        torch.Tensor: Calculated loss
    """
    assert pred.size() == target.size() and target.numel() > 0
    loss = torch.abs(pred - target)**2
    return loss


@LOSSES.register_module()
class L2Loss(nn.Module):
    """L2 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self,
                 neg_pos_ub=-1,
                 pos_margin=-1,
                 neg_margin=-1,
                 hard_mining=False,
                 reduction='mean',
                 loss_weight=1.0):
        super(L2Loss, self).__init__()
        self.neg_pos_ub = neg_pos_ub
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.hard_mining = hard_mining
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        pred, weight, avg_factor = self.update_weight(pred, target, weight,
                                                      avg_factor)
        loss_bbox = self.loss_weight * l2_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_bbox

    def update_weight(self, pred, target, weight, avg_factor):
        if weight is None:
            weight = target.new_ones(target.size())
        invalid_inds = weight <= 0
        target[invalid_inds] = -1
        pos_inds = target == 1
        neg_inds = target == 0

        if self.pos_margin > 0:
            pred[pos_inds] -= self.pos_margin
        if self.neg_margin > 0:
            pred[neg_inds] -= self.neg_margin
        pred = torch.clamp(pred, min=0, max=1)

        num_pos = int((target == 1).sum())
        num_neg = int((target == 0).sum())
        if self.neg_pos_ub > 0 and num_neg / num_pos > self.neg_pos_ub:
            num_neg = num_pos * self.neg_pos_ub
            neg_idx = torch.nonzero(target == 0, as_tuple=False)

            if self.hard_mining:
                costs = l2_loss(
                    pred, target, reduction='none')[neg_idx[:, 0],
                                                    neg_idx[:, 1]].detach()
                neg_idx = neg_idx[costs.topk(num_neg)[1], :]
            else:
                neg_idx = self.random_choice(neg_idx, num_neg)

            new_neg_inds = neg_inds.new_zeros(neg_inds.size()).bool()
            new_neg_inds[neg_idx[:, 0], neg_idx[:, 1]] = True

            invalid_neg_inds = torch.logical_xor(neg_inds, new_neg_inds)
            weight[invalid_neg_inds] = 0

        avg_factor = (weight > 0).sum()
        return pred, weight, avg_factor

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
