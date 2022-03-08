# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
from mmdet.models import HEADS

from mmtrack.core.filter import filter as filter_layer


def softmax_reg_fun(x: torch.Tensor, dim, reg=None):
    """Softmax with optional denominator regularization."""
    if reg is None:
        return torch.softmax(x, dim=dim)
    dim %= x.dim()
    if isinstance(reg, (float, int)):
        reg = x.new_tensor([reg])
    reg = reg.expand([1 if d == dim else x.shape[d] for d in range(x.dim())])
    x = torch.cat((x, reg), dim=dim)
    return torch.softmax(
        x, dim=dim)[[
            slice(-1) if d == dim else slice(None) for d in range(x.dim())
        ]]


@HEADS.register_module()
class PrDiMPSteepestDescentNewton(nn.Module):
    """Optimizer module for PrDiMP.

    It unrolls the steepest descent with Newton iterations to optimize the
        target filter. See the PrDiMP paper.
    args:
        num_iter:  Number of default optimization iterations.
        feat_stride:  The stride of the input feature.
        init_step_length:  Initial scaling of the step length
            (which is then learned).
        init_filter_reg:  Initial filter regularization weight
            (which is then learned).
        gauss_sigma:  The standard deviation to use for the label density
            function.
        min_filter_reg:  Enforce a minimum value on the regularization
            (helps stability sometimes).
        detach_length:  Detach the filter every n-th iteration.
            Default is to never detech, i.e. 'Inf'.
        alpha_eps:  Term in the denominator of the steepest descent that
            stabalizes learning.
        init_uni_weight:  Weight of uniform label distribution.
        normalize_label:  Whether to normalize the label distribution.
        label_shrink:  How much to shrink to label distribution.
        softmax_reg:  Regularization in the denominator of the SoftMax.
        label_threshold:  Threshold probabilities smaller than this.
    """

    def __init__(self,
                 num_iter=1,
                 feat_stride=16,
                 init_step_length=1.0,
                 init_filter_reg=1e-2,
                 gauss_sigma=1.0,
                 min_filter_reg=1e-3,
                 detach_length=float('Inf'),
                 alpha_eps=0.0,
                 init_uni_weight=None,
                 normalize_label=False,
                 label_shrink=0,
                 softmax_reg=None,
                 label_threshold=0.0):
        super().__init__()

        self.num_iter = num_iter
        self.feat_stride = feat_stride
        self.log_step_length = nn.Parameter(
            math.log(init_step_length) * torch.ones(1))
        self.filter_reg = nn.Parameter(init_filter_reg * torch.ones(1))
        self.gauss_sigma = gauss_sigma
        self.min_filter_reg = min_filter_reg
        self.detach_length = detach_length
        self.alpha_eps = alpha_eps
        self.uni_weight = 0 if init_uni_weight is None else init_uni_weight
        self.normalize_label = normalize_label
        self.label_shrink = label_shrink
        self.softmax_reg = softmax_reg
        self.label_threshold = label_threshold

    def get_label_density(self, center, output_sz):
        center = center.reshape(center.shape[0], -1, center.shape[-1])
        k0 = torch.arange(
            output_sz[0], dtype=torch.float32).reshape(1, 1, -1,
                                                       1).to(center.device)
        k1 = torch.arange(
            output_sz[1], dtype=torch.float32).reshape(1, 1, 1,
                                                       -1).to(center.device)
        dist0 = (k0 - center[:, :, 0].reshape(*center.shape[:2], 1, 1))**2
        dist1 = (k1 - center[:, :, 1].reshape(*center.shape[:2], 1, 1))**2
        if self.gauss_sigma == 0:
            dist0_view = dist0.reshape(-1, dist0.shape[-2])
            dist1_view = dist1.reshape(-1, dist1.shape[-1])
            one_hot0 = torch.zeros_like(dist0_view)
            one_hot1 = torch.zeros_like(dist1_view)
            one_hot0[torch.arange(one_hot0.shape[0]),
                     dist0_view.argmin(dim=-1)] = 1.0
            one_hot1[torch.arange(one_hot1.shape[0]),
                     dist1_view.argmin(dim=-1)] = 1.0
            gauss = one_hot0.reshape(dist0.shape) * one_hot1.reshape(
                dist1.shape)
        else:
            g0 = torch.exp(-1.0 / (2 * self.gauss_sigma**2) * dist0)
            g1 = torch.exp(-1.0 / (2 * self.gauss_sigma**2) * dist1)
            gauss = (g0 / (2 * math.pi * self.gauss_sigma**2)) * g1
        gauss = gauss * (gauss > self.label_threshold).float()
        if self.normalize_label:
            gauss /= (gauss.sum(dim=(-2, -1), keepdim=True) + 1e-8)
        label_dens = (1.0 - self.label_shrink) * (
            (1.0 - self.uni_weight) * gauss + self.uni_weight /
            (output_sz[0] * output_sz[1]))
        return label_dens

    def forward(self,
                weights,
                feat,
                bb,
                sample_weight=None,
                num_iter=None,
                compute_losses=True):
        """Runs the optimizer module.

        Note that [] denotes an optional dimension.
        args:
            weights:  Initial weights. Dims (sequences, feat_dim, wH, wW).
            feat:  Input feature maps.
                Dims (images_in_sequence, [sequences], feat_dim, H, W).
            bb:  Target bounding boxes (x, y, w, h) in the image coords.
                Dims (images_in_sequence, [sequences], 4).
            sample_weight:  Optional weight for each sample.
                Dims: (images_in_sequence, [sequences]).
            num_iter:  Number of iterations to run.
            compute_losses:  Whether to compute the (train) loss in each
                iteration.
        returns:
            weights:  The final oprimized weights.
            weight_iterates:  The weights computed in each iteration
                (including initial input and final output).
            losses:  Train losses.
        """

        # Sizes
        num_iter = self.num_iter if num_iter is None else num_iter
        num_images = feat.shape[0]
        num_sequences = feat.shape[1] if feat.dim() == 5 else 1
        filter_sz = (weights.shape[-2], weights.shape[-1])
        output_sz = (feat.shape[-2] + (weights.shape[-2] + 1) % 2,
                     feat.shape[-1] + (weights.shape[-1] + 1) % 2)

        # Get learnable scalars
        step_length_factor = torch.exp(self.log_step_length)
        reg_weight = (self.filter_reg *
                      self.filter_reg).clamp(min=self.min_filter_reg**2)

        # Compute label density
        offset = (torch.Tensor(filter_sz).to(bb.device) % 2) / 2.0
        center = ((bb[..., :2] + bb[..., 2:] / 2) / self.feat_stride).flip(
            (-1, )) - offset
        label_density = self.get_label_density(center, output_sz)

        # Get total sample weights
        if sample_weight is None:
            sample_weight = torch.Tensor([1.0 / num_images]).to(feat.device)
        elif isinstance(sample_weight, torch.Tensor):
            sample_weight = sample_weight.reshape(num_images, num_sequences, 1,
                                                  1)

        exp_reg = 0 if self.softmax_reg is None else math.exp(self.softmax_reg)

        def _compute_loss(scores, weights):
            sample_weight_reshape = sample_weight.reshape(
                sample_weight.shape[0], -1)
            score_log_sum_exp = torch.log(scores.exp().sum(dim=(-2, -1)) +
                                          exp_reg)
            sum_scores = (label_density * scores).sum(dim=(-2, -1))
            return torch.sum(
                sample_weight_reshape * (score_log_sum_exp - sum_scores)
            ) / num_sequences + reg_weight * (weights**2).sum() / num_sequences

        weight_iterates = [weights]
        losses = []

        for i in range(num_iter):
            if i > 0 and i % self.detach_length == 0:
                weights = weights.detach()

            # Compute "residuals"
            scores = filter_layer.apply_filter(feat, weights)
            scores_softmax = softmax_reg_fun(
                scores.reshape(num_images, num_sequences, -1),
                dim=2,
                reg=self.softmax_reg).reshape(scores.shape)
            res = sample_weight * (scores_softmax - label_density)

            if compute_losses:
                losses.append(_compute_loss(scores, weights))

            # Compute gradient
            weights_grad = filter_layer.apply_feat_transpose(
                feat, res, filter_sz,
                training=self.training) + reg_weight * weights

            # Map the gradient with the Hessian
            scores_grad = filter_layer.apply_filter(feat, weights_grad)
            sm_scores_grad = scores_softmax * scores_grad
            hes_scores_grad = sm_scores_grad - scores_softmax * torch.sum(
                sm_scores_grad, dim=(-2, -1), keepdim=True)
            grad_hes_grad = (scores_grad * hes_scores_grad).reshape(
                num_images, num_sequences, -1).sum(dim=2).clamp(min=0)
            grad_hes_grad = (
                sample_weight.reshape(sample_weight.shape[0], -1) *
                grad_hes_grad).sum(dim=0)

            # Compute optimal step length
            alpha_num = (weights_grad * weights_grad).sum(dim=(1, 2, 3))
            alpha_den = (grad_hes_grad +
                         (reg_weight + self.alpha_eps) * alpha_num).clamp(1e-8)
            alpha = alpha_num / alpha_den

            # Update filter
            weights = weights - (step_length_factor *
                                 alpha.reshape(-1, 1, 1, 1)) * weights_grad

            # Add the weight iterate
            weight_iterates.append(weights)

        if compute_losses:
            scores = filter_layer.apply_filter(feat, weights)
            losses.append(_compute_loss(scores, weights))

        return weights, weight_iterates, losses
