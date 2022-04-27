# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn.functional as F
from addict import Dict
from mmdet.models import HEADS
from mmdet.models.builder import build_head
from torch import nn

from mmtrack.core.filter import filter as filter_layer


class InstanceL2Norm(nn.Module):
    """Instance L2 normalization."""

    def __init__(self, size_average=True, eps=1e-5, scale=1.0):
        super().__init__()
        self.size_average = size_average
        self.eps = eps
        self.scale = scale

    def forward(self, input):
        if self.size_average:
            return input * (
                self.scale *
                ((input.shape[1] * input.shape[2] * input.shape[3]) /
                 (torch.sum(
                     (input * input).view(input.shape[0], 1, 1, -1),
                     dim=3,
                     keepdim=True) + self.eps)).sqrt())
        else:
            return input * (
                self.scale / (torch.sum(
                    (input * input).view(input.shape[0], 1, 1, -1),
                    dim=3,
                    keepdim=True) + self.eps).sqrt())


# def softmax_reg(x: torch.Tensor, dim, reg=None):
#     """Softmax with optional denominator regularization."""
#     if reg is None:
#         return torch.softmax(x, dim=dim)
#     dim %= x.dim()
#     if isinstance(reg, (float, int)):
#         reg = x.new_tensor([reg])
#     reg = reg.expand([1 if d == dim else x.shape[d] for d in range(x.dim())])
#     x = torch.cat((x, reg), dim=dim)
#     return torch.softmax(
#         x, dim=dim)[[
#             slice(-1) if d == dim else slice(None) for d in range(x.dim())
#         ]]


def max2d(a: torch.Tensor):
    """Computes maximum and argmax in the last two dimensions."""

    max_val_row, argmax_row = torch.max(a, dim=-2)
    max_val, argmax_col = torch.max(max_val_row, dim=-1)
    argmax_row = argmax_row.view(argmax_col.numel(),
                                 -1)[torch.arange(argmax_col.numel()),
                                     argmax_col.view(-1)]
    argmax_row = argmax_row.reshape(argmax_col.shape)
    argmax = torch.cat((argmax_row.unsqueeze(-1), argmax_col.unsqueeze(-1)),
                       -1)
    return max_val, argmax


@HEADS.register_module()
class PrdimpClsHead(nn.Module):

    def __init__(self,
                 in_dim=1024,
                 out_dim=512,
                 filter_initializer=None,
                 filter_optimizer=None,
                 locate_cfg=None,
                 update_cfg=None,
                 optimizer_cfg=None,
                 loss_cls=None,
                 train_cfg=None,
                 **kwargs):
        super().__init__()
        filter_size = filter_initializer.filter_size
        self.filter_initializer = build_head(filter_initializer)
        self.filter_optimizer = build_head(filter_optimizer)
        norm_scale = math.sqrt(1.0 / (out_dim * filter_size * filter_size))
        self.cls_feature_extractor = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, bias=False),
            InstanceL2Norm(scale=norm_scale))

        self.locate_cfg = locate_cfg
        self.update_cfg = update_cfg
        self.optimizer_cfg = optimizer_cfg

        if isinstance(filter_size, (int, float)):
            filter_size = [filter_size, filter_size]
        self.filter_size = torch.tensor(filter_size, dtype=torch.float32)

    def init_weights(self):
        for m in self.cls_feature_extractor.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.filter_initializer.init_weights()

    def init_classifier(self,
                        backbone_feats,
                        target_bboxes,
                        dropout_probs=None):
        """Initialize the filter and memory in the classifier.

        Args:
            backbone_feats (Tensor): _description_
            target_bboxes (Tensor): _description_
            dropout_probs (List, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        with torch.no_grad():
            cls_feats = self.cls_feature_extractor(backbone_feats)
        # add features through the augmentation of `dropout`
        if dropout_probs is not None:
            aug_feats = []
            for i, prob in enumerate(dropout_probs):
                aug_feat = F.dropout2d(
                    cls_feats[:1, ...], p=prob, training=True)
                aug_feats.append(aug_feat)
            cls_feats = torch.cat([cls_feats] + aug_feats)

        # Get target filter by running the discriminative model prediction
        # module
        with torch.no_grad():
            self.target_filter, _, _ = self.get_filter(
                cls_feats,
                target_bboxes,
                num_iter=self.optimizer_cfg['net_opt_iter'])

        # Initialize memory
        self.init_memory(cls_feats, target_bboxes)

        return list(cls_feats.shape[-2:])

    def localize_target(self, scores, sample_pos, sample_scale, sample_size,
                        prev_bbox_hw, prev_bbox_yx):
        """Run the target advanced localization (as in ATOM).

        scores (Tensor): of shape (1, h, w) sample_pos (Tensor): of shape (1,
        2) sample_scale (Tensor): of shape (1,) target_scale is the ratio of
        the     size of cropped image to the size of resized image. sample_size
        (Tensor): of shape (2,), the size of input size prev_bbox_hw (Tensor):
        of shape (2,) prev_bbox_yx (Tensor): of shape (2,)
        """
        scores = scores.squeeze()
        sz = scores.shape
        score_sz = torch.tensor(list(sz))
        output_sz = score_sz - (self.filter_size + 1) % 2
        score_center = (score_sz - 1) / 2

        max_score1, max_disp1 = max2d(scores)
        # _, scale_ind = torch.max(max_score1, dim=0)
        # sample_scale = sample_scales[scale_ind]
        # max_score1 = max_score1[scale_ind]
        # max_disp1 = max_disp1[scale_ind, ...].float().cpu().view(-1)
        max_disp1 = max_disp1.float().cpu()
        target_disp1 = max_disp1 - score_center
        translation_vec1 = target_disp1 * (sample_size /
                                           output_sz) * sample_scale

        if max_score1.item() < self.locate_cfg['target_not_found_threshold']:
            return translation_vec1, 'not_found'
        if max_score1.item() < self.locate_cfg.get('uncertain_threshold',
                                                   -float('inf')):
            return translation_vec1, 'uncertain'
        if max_score1.item() < self.locate_cfg.get('hard_sample_threshold',
                                                   -float('inf')):
            return translation_vec1, 'hard_negative'

        # Mask out target neighborhood
        target_neigh_sz = self.locate_cfg['target_neighborhood_scale'] * (
            prev_bbox_hw / sample_scale) * (
                output_sz / sample_size)

        tneigh_top = max(
            round(max_disp1[0].item() - target_neigh_sz[0].item() / 2), 0)
        tneigh_bottom = min(
            round(max_disp1[0].item() + target_neigh_sz[0].item() / 2 + 1),
            sz[0])
        tneigh_left = max(
            round(max_disp1[1].item() - target_neigh_sz[1].item() / 2), 0)
        tneigh_right = min(
            round(max_disp1[1].item() + target_neigh_sz[1].item() / 2 + 1),
            sz[1])
        scores_masked = scores.clone()
        scores_masked[tneigh_top:tneigh_bottom, tneigh_left:tneigh_right] = 0

        # Find new maximum
        max_score2, max_disp2 = max2d(scores_masked)
        max_disp2 = max_disp2.float().cpu().view(-1)
        target_disp2 = max_disp2 - score_center
        translation_vec2 = target_disp2 * (sample_size /
                                           output_sz) * sample_scale

        prev_target_vec = (prev_bbox_yx - sample_pos[0, :]) / (
            (sample_size / output_sz) * sample_scale)

        # Handle the different cases
        if max_score2 > self.locate_cfg['distractor_threshold'] * max_score1:
            disp_norm1 = torch.sqrt(
                torch.sum((target_disp1 - prev_target_vec)**2))
            disp_norm2 = torch.sqrt(
                torch.sum((target_disp2 - prev_target_vec)**2))
            disp_threshold = self.locate_cfg['dispalcement_scale'] * math.sqrt(
                sz[0] * sz[1]) / 2

            if disp_norm2 > disp_threshold and disp_norm1 < disp_threshold:
                return translation_vec1, 'hard_negative'
            if disp_norm2 < disp_threshold and disp_norm1 > disp_threshold:
                return translation_vec2, 'hard_negative'
            if disp_norm2 > disp_threshold and disp_norm1 > disp_threshold:
                return translation_vec1, 'uncertain'

            # If also the distractor is close, return with highest score
            return translation_vec1, 'uncertain'

        if (max_score2 >
                self.locate_cfg['hard_negative_threshold'] * max_score1 and
                max_score2 > self.locate_cfg['target_not_found_threshold']):
            return translation_vec1, 'hard_negative'

        return translation_vec1, 'normal'

    def init_memory(self, aug_feats, target_bboxes):
        """Initialize the some properties about training samples in memory:

            - `bboxes` (N, 4): the gt_bboxes of all samples
            - `training_samples` (N, C, H, W): the features of training samples
            - `sample_weights` (N,): the weights of all samples
            - `num_samples` (int): the number of all the samples fed into
                memory, including the outdated samples.
            - `replace_ind` (int): the index of samples in memory which would
                be replaced by the next new samples.

        Args:
            aug_feats (_type_): _description_
            target_bboxes (_type_): _description_
        """
        self.memo = Dict()

        self.memo.bboxes = target_bboxes.new_zeros(
            self.update_cfg['sample_memory_size'], 4)
        self.memo.bboxes[:target_bboxes.shape[0], :] = target_bboxes

        # Initialize first-frame spatial training samples
        self.memo.num_samples = self.num_init_samples = aug_feats.size(0)
        # the index of the replaced memory samples in the next frame
        self.memo.replace_ind = None

        self.memo.sample_weights = aug_feats.new_zeros(
            self.update_cfg['sample_memory_size'])
        self.memo.sample_weights[:self.num_init_samples] = aug_feats.new_ones(
            1) / aug_feats.shape[0]

        self.memo.training_samples = aug_feats.new_zeros(
            self.update_cfg['sample_memory_size'], *aug_feats.shape[1:])
        self.memo.training_samples[:self.num_init_samples] = aug_feats

    def update_memory(self, sample_x, target_box, learning_rate=None):
        # Update weights and get replace ind
        replace_ind = self.update_sample_weights(learning_rate)
        self.memo.replace_ind = replace_ind

        # Update training samples and bboxes in memory
        self.memo.training_samples[replace_ind:replace_ind + 1, ...] = sample_x
        self.memo.bboxes[replace_ind, :] = target_box
        self.memo.num_samples += 1

    def update_sample_weights(self, learning_rate=None):
        """Update the weights of samples in memory.

        Args:
            learning_rate (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """

        if learning_rate is None:
            learning_rate = self.locate_cfg.learning_rate

        init_sample_weight = self.locate_cfg.get('init_samples_minimum_weight',
                                                 None)
        if init_sample_weight == 0:
            init_sample_weight = None

        replace_start_ind = 0 if init_sample_weight is None else \
            self.num_init_samples

        if self.memo.num_samples == 0 or learning_rate == 1:
            self.memo.sample_weights[:] = 0
            self.memo.sample_weights[0] = 1
            replace_ind = 0
        else:
            # Get index to replace
            if self.memo.num_samples < self.memo.sample_weights.shape[0]:
                replace_ind = self.memo.num_samples
            else:
                _, replace_ind = torch.min(
                    self.memo.sample_weights[replace_start_ind:], 0)
                replace_ind = replace_ind.item() + replace_start_ind

            # Update weights
            if self.memo.replace_ind is None:
                self.memo.sample_weights /= 1 - learning_rate
                self.memo.sample_weights[replace_ind] = learning_rate
            else:
                self.memo.sample_weights[
                    replace_ind] = self.memo.sample_weights[
                        self.memo.replace_ind] / (1 - learning_rate)

        self.memo.sample_weights /= self.memo.sample_weights.sum()
        if (init_sample_weight is not None
                and self.memo.sample_weights[:self.num_init_samples].sum() <
                init_sample_weight):
            # TODO werid! the sum of samples_weights is not equal to 1.
            self.memo.sample_weights /= (
                init_sample_weight +
                self.memo.sample_weights[self.num_init_samples:].sum())
            self.memo.sample_weights[:self.num_init_samples] = (
                init_sample_weight / self.num_init_samples)

        return replace_ind

    def update_classifier(self,
                          train_x,
                          target_box,
                          frame_num,
                          learning_rate=None,
                          scores=None):
        # Set flags and learning rate
        hard_negative_flag = learning_rate is not None
        if learning_rate is None:
            learning_rate = self.update_cfg['learning_rate']

        # Update the tracker memory
        if hard_negative_flag or frame_num % self.locate_cfg.get(
                'train_sample_interval', 1) == 0:
            self.update_memory(train_x, target_box, learning_rate)

        # Decide the number of iterations to run
        num_iter = 0
        low_score_th = self.optimizer_cfg.get('low_score_opt_threshold', None)
        if hard_negative_flag:
            num_iter = self.optimizer_cfg.get('net_opt_hn_iter', None)
        elif low_score_th is not None and low_score_th > scores.max().item():
            num_iter = self.optimizer_cfg.get('net_opt_low_iter', None)
        elif (frame_num - 1) % self.update_cfg['train_skipping'] == 0:
            num_iter = self.optimizer_cfg.get('net_opt_update_iter', None)

        if num_iter > 0:
            # Get inputs for the DiMP filter optimizer module
            samples = self.memo.training_samples[:self.memo.num_samples, ...]
            target_boxes = self.memo.bboxes[:self.memo.num_samples, :].clone()
            sample_weights = self.memo.sample_weights[:self.memo.num_samples]

            # Run the filter optimizer module
            with torch.no_grad():
                self.target_filter, _, losses = self.filter_optimizer(
                    self.target_filter,
                    num_iter=num_iter,
                    feat=samples,
                    bb=target_boxes,
                    sample_weight=sample_weights)

    def classify(self, feat, filter=None):
        """Run classifier on the features."""
        if filter is None:
            scores = filter_layer.apply_filter(feat, self.target_filter)
        else:
            assert feat[0].shape[:2] == filter.shape[:2]
            scores = filter_layer.apply_filter(feat, filter)

        return scores

    def get_filter(self, feat, bb, *args, **kwargs):
        """Outputs the learned filter based on the input features (feat) and
            target boxes (bb) by running the filter initializer and optimizer.
            Note that [] denotes an optional dimension.
        args:
            feat:  Input feature maps. Dims
                (images_in_sequence, [sequences], feat_dim, H, W).
            bb:  Target bounding boxes (x, y, w, h) in the image coords.
                Dims (images_in_sequence, [sequences], 4).
            *args, **kwargs:  These are passed to the optimizer module.
        returns:
            weights:  The final oprimized weights.
                Dims (sequences, feat_dim, wH, wW).
            weight_iterates:  The weights computed in each iteration
                (including initial input and final output).
            losses:  Train losses."""

        weights = self.filter_initializer(feat, bb)

        if self.filter_optimizer is not None:
            weights, weights_iter, losses = self.filter_optimizer(
                weights, feat=feat, bb=bb, *args, **kwargs)
        else:
            weights_iter = [weights]
            losses = None

        return weights, weights_iter, losses
