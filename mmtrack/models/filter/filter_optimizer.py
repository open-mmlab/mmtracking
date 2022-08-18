# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from mmengine.model import BaseModule
from torch import Tensor

from mmtrack.registry import MODELS
from ..task_modules.filter import filter as filter_layer


@MODELS.register_module()
class PrDiMPFilterOptimizer(BaseModule):
    """Optimizer module of filter in PrDiMP.

    It unrolls the steepest descent with Newton iterations to optimize the
        target filter.

    Args:
        num_iters (int, optional):  Number of default optimization iterations.
            Defaults to 1.
        feat_stride (int, optional):  The stride of the input feature.
            Defaults to 16.
        init_step_length (float, optional):  Initial scaling of the step length
            (which is then learned). Defaults to 1.0.
        init_filter_regular (float, optional):  Initial filter regularization
            weight (which is then learned). Defaults to 1e-2.
        gauss_sigma (float, optional):  The standard deviation to use for the
            label density function. Defaults to 1.0.
        min_filter_regular (float, optional):  Enforce a minimum value on the
            regularization (helps stability sometimes). Defaults to 1e-3.
        alpha_eps (float, optional):  Term in the denominator of the steepest
            descent that stabalizes learning. Defaults to 0.
        label_thres (float, optional):  Threshold probabilities. Defaults to 0.
    """

    def __init__(self,
                 num_iters: int = 1,
                 feat_stride: int = 16,
                 init_step_length: float = 1.0,
                 init_filter_regular: float = 1e-2,
                 gauss_sigma: float = 1.0,
                 min_filter_regular: float = 1e-3,
                 alpha_eps: float = 0.0,
                 label_thres: float = 0.0):
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

    def gen_label_density(self, center_yx: Tensor,
                          output_size_hw: Tensor) -> Tensor:
        """Generate label density.

        Args:
            center_yx (Tensor): The center of score map.
            output_size_hw (Tensor): The size of score map in [h, w] format.

        Returns:
            Tensor: Label density with two possible shape:
                - train mode: (num_img_per_seq, bs, h, w).
                - test mode: (num_img_per_seq, 1, h, w).
        """
        # convert to (num_img_per_seq, bs, 4) shape
        center_yx = center_yx.reshape(center_yx.shape[0], -1,
                                      center_yx.shape[-1])
        k0 = torch.arange(
            output_size_hw[0],
            dtype=torch.float32).reshape(1, 1, -1, 1).to(center_yx.device)
        k1 = torch.arange(
            output_size_hw[1],
            dtype=torch.float32).reshape(1, 1, 1, -1).to(center_yx.device)
        dist0 = (k0 -
                 center_yx[:, :, 0].reshape(*center_yx.shape[:2], 1, 1))**2
        dist1 = (k1 -
                 center_yx[:, :, 1].reshape(*center_yx.shape[:2], 1, 1))**2
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
                filter_weights: Tensor,
                feat: Tensor,
                bboxes: Tensor,
                num_iters: Optional[int] = None,
                sample_weights: Optional[Tensor] = None) -> Tuple[Tensor, ...]:
        """Runs the optimizer module.

        Note that [] denotes an optional dimension. Generally speaking, inputs
        in test mode don't have the dim of [].

        Args:
            filter_weights (Tensor):  Initial filter with shape
                training mode: (bs, c, fitler_h, filter_w)
                test mode: (1, c, fitler_h, filter_w)
            feat (Tensor):  Input feature maps with shape
                (num_img_per_seq, [bs], c, H, W).
            bboxes (Tensor):  Target bounding boxes with shape
                (num_img_per_seq, [bs], 4). in (cx, cy, x, y) format.
            num_iters (int, optional):  Number of iterations to run.
                Defaults to None.
            sample_weights (Tensor, optional):  Optional weight for each
                sample with shape (num_img_per_seq, [bs]). Defaults to None.

        Returns:
            filter_weights (Tensor):  The final oprimized filter.
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
        filter_size_hw = (filter_weights.shape[-2], filter_weights.shape[-1])
        output_size_hw = (feat.shape[-2] + (filter_weights.shape[-2] + 1) % 2,
                          feat.shape[-1] + (filter_weights.shape[-1] + 1) % 2)

        # Get learnable scalars
        step_length_factor = torch.exp(self.log_step_length)
        filter_regular = (self.filter_regular**2).clamp(
            min=self.min_filter_regular**2)

        # Compute label density
        if self.training:
            assert bboxes.dim() == 3
        else:
            assert bboxes.dim() == 2
            bboxes = bboxes.reshape([bboxes.shape[0], -1, bboxes.shape[-1]])
        offset = (torch.Tensor(filter_size_hw).to(bboxes.device) % 2) / 2.0
        center = bboxes[..., :2] / self.feat_stride
        center_yx = center.flip((-1, )) - offset
        label_density = self.gen_label_density(center_yx, output_size_hw)

        # Get total sample weights
        if sample_weights is None:
            sample_weights = torch.Tensor([1.0 / num_img_per_seq
                                           ]).to(feat.device)
        elif isinstance(sample_weights, torch.Tensor):
            sample_weights = sample_weights.reshape(num_img_per_seq,
                                                    batch_size, 1, 1)
        else:
            raise NotImplementedError(
                "Only support two types of 'sample_weights': "
                'torch.Tensor or None')

        filter_iters = []
        losses = []

        for _ in range(num_iters):
            # Get scores by applying the filter on the features
            scores = filter_layer.apply_filter(feat, filter_weights)
            scores = torch.softmax(
                scores.reshape(num_img_per_seq, batch_size, -1),
                dim=2).reshape(scores.shape)

            # Compute loss and record the filter of each iteration in training
            # mode.
            if self.training:
                filter_iters.append(filter_weights)
                losses.append(
                    self._compute_loss(scores, sample_weights, label_density,
                                       filter_weights, filter_regular))

            # Compute gradient and step_length
            res = sample_weights * (scores - label_density)
            filter_grad = filter_layer.apply_feat_transpose(
                feat, res, filter_size_hw,
                training=self.training) + filter_regular * filter_weights

            step_length = self.get_step_length(feat, sample_weights, scores,
                                               filter_grad, filter_regular)

            # Update filter
            filter_weights = filter_weights - (
                step_length_factor *
                step_length.reshape(-1, 1, 1, 1)) * filter_grad

        if self.training:
            filter_iters.append(filter_weights)
            # Get scores by applying the final filter on the feature map
            scores = filter_layer.apply_filter(feat, filter_weights)
            losses.append(
                self._compute_loss(scores, sample_weights, label_density,
                                   filter_weights, filter_regular))
            return filter_weights, filter_iters, losses
        else:
            return filter_weights

    def get_step_length(self, feat: Tensor, sample_weights: Tensor,
                        scores: Tensor, filter_grad: Tensor,
                        filter_regular: Tensor) -> Tensor:
        """Compute the step length of updating the filter.

        Args:
            feat (Tensor): Input feature map with shape
                (num_img_per_seq, [bs], feat_dim, H, W).
            sample_weights (Tensor): The weights of all the samples.
            scores (Tensor): The score map with two possible shape:
                - train mode: (num_img_per_seq, bs, h, w).
                - test mode: (num_img_per_seq, 1, h, w).
            filter_grad (Tensor): The gradient of the filter with shape
                (num_img_per_seq, c, fitler_h, filter_w).
            filter_regular (Tensor): The regulazation item of the filter, with
                shape (1,).

        Returns:
            alpha (Tensor): The updating factor with shape (1, ).
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

    def _compute_loss(self, scores: Tensor, sample_weights: Tensor,
                      label_density: Tensor, filter: Tensor,
                      filter_regular: Tensor) -> Tensor:
        """Compute loss in the box optimization.

        Args:
            scores (Tensor): The score map with shape
                (num_img_per_seq, bs, h, w).
            sample_weights (Tensor): The weights of all the samples with shape
                (num_img_per_seq, bs, 1, 1)
            label_density (Tensor):The label density with shape
                (num_img_per_seq, bs, h, w).
            filter (Tensor): The filter with shape
                (num_img_per_seq, c, fitler_h, filter_w).
            filter_regular (Tensor):The regulazation item of the filter, with
                shape (1,).

        Returns:
            Tensor: with shape (1,)
        """

        num_samples = sample_weights.shape[0]
        sample_weights = sample_weights.reshape(sample_weights.shape[0], -1)
        score_log_sum_exp = torch.log(scores.exp().sum(dim=(-2, -1)))
        sum_scores = (label_density * scores).sum(dim=(-2, -1))

        return torch.sum(
            sample_weights * (score_log_sum_exp - sum_scores)
        ) / num_samples + filter_regular * (filter**2).sum() / num_samples
