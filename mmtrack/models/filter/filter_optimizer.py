# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
from mmdet.models import HEADS

from mmtrack.core.filter import filter as filter_layer


@HEADS.register_module()
class PrDiMPSteepestDescentNewton(nn.Module):
    """Optimizer module for PrDiMP.

    It unrolls the steepest descent with Newton iterations to optimize the
        target filter. It's used on PrDiMP.

    Args:
        num_iters:  Number of default optimization iterations.
        feat_stride:  The stride of the input feature.
        init_step_length:  Initial scaling of the step length
            (which is then learned).
        init_filter_regular:  Initial filter regularization weight
            (which is then learned).
        gauss_sigma:  The standard deviation to use for the label density
            function.
        min_filter_regular:  Enforce a minimum value on the regularization
            (helps stability sometimes).
        alpha_eps:  Term in the denominator of the steepest descent that
            stabalizes learning.
        label_thres:  Threshold probabilities smaller than this.
    """

    def __init__(self,
                 num_iters=1,
                 feat_stride=16,
                 init_step_length=1.0,
                 init_filter_regular=1e-2,
                 gauss_sigma=1.0,
                 min_filter_regular=1e-3,
                 alpha_eps=0.0,
                 label_thres=0.0):
        super().__init__()

        self.num_iters = num_iters
        self.feat_stride = feat_stride
        self.log_step_length = nn.Parameter(
            math.log(init_step_length) * torch.ones(1))
        self.filter_regular = nn.Parameter(init_filter_regular * torch.ones(1))
        self.gauss_sigma = gauss_sigma
        self.min_filter_regular = min_filter_regular
        self.alpha_eps = alpha_eps
        self.label_thres = label_thres

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
        gauss = gauss * (gauss > self.label_thres).float()
        gauss_density = gauss / (gauss.sum(dim=(-2, -1), keepdim=True) + 1e-8)
        return gauss_density

    def forward(self,
                filter,
                feat,
                bboxes,
                num_iters=None,
                sample_weights=None):
        """Runs the optimizer module.

        Note that [] denotes an optional dimension. Generally speaking, inputs
        in test mode don't have the dim of [].

        args:
            filter:  Initial filter with shape
                (num_img_per_seq, c, fitler_h, filter_w).
            feat:  Input feature maps with shape
                (num_img_per_seq, [bs], feat_dim, H, W).
            bboxes:  Target bounding boxes with shape
                (num_img_per_seq, [bs], 4). in (x, y, w, h) format.
            sample_weights:  Optional weight for each sample.
                Dims: (num_img_per_seq, [bs]).
            num_iters:  Number of iterations to run.

        returns:
            filter (Tensor):  The final oprimized filter.
            filter_iters (Tensor, optional):  The filter computed in each
                iteration (including initial input and final output), returned
                only in training
            losses (Tensor, optional): losses in all optimizer iterations,
                returned only in training
        """

        # Sizes
        num_iters = self.num_iters if num_iters is None else num_iters
        num_img_per_seq = feat.shape[0]
        batch_size = feat.shape[1] if feat.dim() == 5 else 1
        filter_sz = (filter.shape[-2], filter.shape[-1])
        output_sz = (feat.shape[-2] + (filter.shape[-2] + 1) % 2,
                     feat.shape[-1] + (filter.shape[-1] + 1) % 2)

        # Get learnable scalars
        step_length_factor = torch.exp(self.log_step_length)
        filter_regular = (self.filter_regular**2).clamp(
            min=self.min_filter_regular**2)

        # Compute label density
        offset = (torch.Tensor(filter_sz).to(bboxes.device) % 2) / 2.0
        center = ((bboxes[..., :2] + bboxes[..., 2:] / 2) / self.feat_stride)
        center = center.flip((-1, )) - offset
        label_density = self.get_label_density(center, output_sz)

        # Get total sample filter
        if sample_weights is None:
            sample_weights = torch.Tensor([1.0 / num_img_per_seq
                                           ]).to(feat.device)
        elif isinstance(sample_weights, torch.Tensor):
            sample_weights = sample_weights.reshape(num_img_per_seq,
                                                    batch_size, 1, 1)

        filter_iters = []
        losses = []

        for i in range(num_iters):
            # Get scores by applying the filter on the features
            scores = filter_layer.apply_filter(feat, filter)
            scores = torch.softmax(
                scores.reshape(num_img_per_seq, batch_size, -1),
                dim=2).reshape(scores.shape)

            # Compute loss and record the filter of each iteration in training
            # mode.
            if self.training:
                filter_iters.append(filter)
                losses.append(
                    self._compute_loss(scores, sample_weights, label_density,
                                       filter, filter_regular))

            # Compute gradient and step_length
            res = sample_weights * (scores - label_density)
            filter_grad = filter_layer.apply_feat_transpose(
                feat, res, filter_sz,
                training=self.training) + filter_regular * filter

            step_length = self.get_step_length(feat, sample_weights, scores,
                                               filter_grad, filter_regular)

            # Update filter
            filter = filter - (step_length_factor *
                               step_length.reshape(-1, 1, 1, 1)) * filter_grad

        if self.training:
            filter_iters.append(filter)
            # Get scores by applying the final filter on the feature map
            scores = filter_layer.apply_filter(feat, filter)
            losses.append(
                self._compute_loss(scores, sample_weights, label_density,
                                   filter, filter_regular))
            return filter, filter_iters, losses
        else:
            return filter

    def get_step_length(self, feat, sample_weights, scores, filter_grad,
                        filter_regular):
        """Compute the step length of updating the filter.

        Args:
            feat (Tensor): Input feature map with shape.
            sample_weights (Tensor): The weights of all samples with shape ()
            scores (Tensor): The score map with shaoe ().
            filter_grad (Tensor): The gradient of the filter with shape ().
            filter_regular (Tensor): The regulazation item of the filter, with
                shape ().

        Returns:
            alpha (Tensor): The updating factor with shape ().
        """
        num_img_per_seq = feat.shape[0]
        batch_size = feat.shape[1] if feat.dim() == 5 else 1
        # Map the gradient with the Hessian
        scores_grad = filter_layer.apply_filter(feat, filter_grad)
        sm_scores_grad = scores * scores_grad
        hes_scores_grad = sm_scores_grad - scores * torch.sum(
            sm_scores_grad, dim=(-2, -1), keepdim=True)
        grad_hes_grad = (scores_grad * hes_scores_grad).reshape(
            num_img_per_seq, batch_size, -1).sum(dim=2).clamp(min=0)
        grad_hes_grad = (sample_weights.reshape(sample_weights.shape[0], -1) *
                         grad_hes_grad).sum(dim=0)

        # Compute optimal step length
        alpha_num = (filter_grad * filter_grad).sum(dim=(1, 2, 3))
        alpha_den = (grad_hes_grad +
                     (filter_regular + self.alpha_eps) * alpha_num).clamp(1e-8)
        alpha = alpha_num / alpha_den

        return alpha

    def _compute_loss(scores, sample_weights, label_density, filter,
                      filter_regular):
        """Compute loss in the box optimization.

        Args:
            scores (Tensor): _description_
            sample_weights (Tensor): _description_
            label_density (Tensor): _description_
            filter (Tensor): _description_
            filter_regular (Tensor): _description_

        Returns:
            (Tensor):
        """

        num_samples = sample_weights.shape[0]
        sample_weights = sample_weights.reshape(sample_weights.shape[0], -1)
        score_log_sum_exp = torch.log(scores.exp().sum(dim=(-2, -1)))
        sum_scores = (label_density * scores).sum(dim=(-2, -1))

        return torch.sum(
            sample_weights * (score_log_sum_exp - sum_scores)
        ) / num_samples + filter_regular * (filter**2).sum() / num_samples
