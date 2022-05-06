# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn.functional as F
from addict import Dict
from mmdet.core.bbox.transforms import bbox_cxcywh_to_xyxy
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
            backbone_feats (Tensor): The features from backbone.
            target_bboxes (Tensor): in [cx, cy, w, h] format.
            dropout_probs (List, optional): Defaults to None.

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
        target_bboxes_xyxy = bbox_cxcywh_to_xyxy(target_bboxes)
        with torch.no_grad():
            init_filter = self.filter_initializer(cls_feats,
                                                  target_bboxes_xyxy)
            self.target_filter = self.filter_optimizer(
                init_filter,
                feat=cls_feats,
                bboxes=target_bboxes,
                num_iters=self.optimizer_cfg['net_opt_iter'])

        # Initialize memory
        self.init_memory(cls_feats, target_bboxes)

        return [cls_feats.shape[-1], cls_feats.shape[-2]]

    def init_memory(self, aug_feats, target_bboxes):
        """Initialize the some properties about training samples in memory:

            - `bboxes` (N, 4): the gt_bboxes of all samples in [cx, cy, w, h]
                format.
            - `training_samples` (N, C, H, W): the features of training samples
            - `sample_weights` (N,): the weights of all samples
            - `num_samples` (int): the number of all the samples fed into
                memory, including the outdated samples.
            - `replace_ind` (int): the index of samples in memory which would
                be replaced by the next new samples.

        Args:
            aug_feats (Tensor): The augmented features.
            target_bboxes (Tensor): of shape (N, 4) in [cx, cy, w, h] format.
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

    def localize_target(self, scores, sample_center, sample_scale, sample_size,
                        prev_bbox):
        """Run the target localization based on the score map.

        Args:
            scores (Tensor): It's of shape (1, h, w).
            sample_center (Tensor): The center of the cropped
                sample on the original image. It's of shape (1,2) in [x, y]
                format.
            sample_scale (Tensor): The scale of the cropped sample. It's of
                shape (1,).
            sample_size (Tensor): The scale of the cropped sample. It's of
                shape (2,) in [h, w] format.
            prev_bbox (Tensor): It's of shape (4,) in [cx, cy, w, h] format.

        Return:
            Tensor: The displacement of the target to the center of original
                image
            bool: The tracking state
        """
        scores = scores.squeeze()
        score_size = torch.tensor([scores.shape[-1], scores.shape[-2]])
        output_size = (score_size - (self.filter_size + 1) % 2).flip(0).to(
            sample_size.device)
        score_center = ((score_size - 1) / 2).to(scores.device)

        max_score, max_pos = max2d(scores)
        max_pos = max_pos.flip(0).float()
        # the displacement of target to the center of score map
        target_disp_score_map = max_pos - score_center
        # the ratio of the size of original image to to that of score map
        ratio_size = (sample_size / output_size) * sample_scale
        # the displcement of the target to the center of original image
        target_disp = target_disp_score_map * ratio_size

        # Handle different cases
        # 1. Target is not found
        if max_score.item() < self.locate_cfg['no_target_min_score']:
            return target_disp, 'not_found'

        # Analysis whether there is a distractor
        # Calculate the size of neighborhood near the current target
        target_neigh_sz = self.locate_cfg['target_neighborhood_scale'] * (
            prev_bbox[2:4] / ratio_size)

        top_left = (max_pos - target_neigh_sz / 2).round().long()
        top_left = torch.clamp_min(top_left, 0).tolist()
        bottom_right = (max_pos + target_neigh_sz / 2 + 1).round().long()
        bottom_right = torch.clamp_max(bottom_right, score_size.min()).tolist()
        scores_masked = scores.clone()
        scores_masked[top_left[1]:bottom_right[1],
                      top_left[0]:bottom_right[0]] = 0

        # Find new maximum except the neighborhood of the target
        second_max_score, second_max_pos = max2d(scores_masked)
        second_max_pos = second_max_pos.flip(0).float().view(-1)
        distractor_disp_score_map = second_max_pos - score_center
        distractor_disp = distractor_disp_score_map * ratio_size
        # The displacement of previout target bbox to the center of the score
        # map
        # TODO: check it
        prev_target_disp_score_map = (prev_bbox[:2] -
                                      sample_center[0, :]) / ratio_size

        # 2. There is a distractor
        if second_max_score > self.locate_cfg['distractor_thres'] * max_score:
            target_disp_diff = torch.sqrt(
                torch.sum(
                    (target_disp_score_map - prev_target_disp_score_map)**2))
            distractor_disp_diff = torch.sqrt(
                torch.sum((distractor_disp_score_map -
                           prev_target_disp_score_map)**2))
            disp_diff_thres = self.locate_cfg[
                'dispalcement_scale'] * score_size.prod().float().sqrt() / 2

            if (distractor_disp_diff > disp_diff_thres
                    and target_disp_diff < disp_diff_thres):
                return target_disp, 'hard_negative'
            if (distractor_disp_diff < disp_diff_thres
                    and target_disp_diff > disp_diff_thres):
                # The true target may be the `distractor` instead of the
                # `target` on this frame
                return distractor_disp, 'hard_negative'
            else:
                # If both the displacement of target and distractor is larger
                # or smaller than the threshold, return the displacement of the
                # highest score.
                return target_disp, 'uncertain'

        # 3. There is a hard negative object
        if (second_max_score > self.locate_cfg['hard_neg_thres'] * max_score
                and second_max_score > self.locate_cfg['no_target_min_score']):
            return target_disp, 'hard_negative'

        # 4. Normal target
        return target_disp, 'normal'

    def classify(self, feat):
        """Run classifier on the features."""
        scores = filter_layer.apply_filter(feat, self.target_filter)
        return scores

    def update_memory(self, sample_x, target_bbox, learning_rate=None):
        """Update the tracking state in memory.

        Args:
            sample_x (Tensor): The feature from backbone.
            target_bbox (Tensor): of shape (1,4) in [x, y, w, h] format.
            learning_rate (float, optional): The learning rate about updating.
                Defaults to None.
        """
        # Update weights and get replace ind
        replace_ind = self.update_sample_weights(learning_rate)
        self.memo.replace_ind = replace_ind

        # Update training samples and bboxes in memory
        self.memo.training_samples[replace_ind:replace_ind + 1, ...] = sample_x
        self.memo.bboxes[replace_ind, :] = target_bbox
        self.memo.num_samples += 1

    def update_sample_weights(self, learning_rate=None):
        """Update the weights of samples in memory.

        Args:
            learning_rate (int, optional): The learning rate of updating
                samples in memory. Defaults to None.

        Returns:
            (int): the index of updated samples in memory.
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
                          train_feat,
                          target_bbox,
                          frame_num,
                          hard_neg_flag=False):
        """Update the classifier with the refined bbox.

        Args:
            train_feat (Tensor): The feature from backbone.
            target_bbox (Tensor): of shape (1, 4) in [x, y, w, h] format.
            frame_num (int): The ID of frame.
            hard_neg_flag (bool, optional): Whether is the hard negative
                sample. Defaults to False.
        """
        # Set flags and learning rate
        learning_rate = self.update_cfg[
            'normal_lr'] if not hard_neg_flag else self.update_cfg[
                'hard_neg_lr']

        # Update the tracker memory
        if hard_neg_flag or frame_num % self.locate_cfg.get(
                'train_sample_interval', 1) == 0:
            self.update_memory(train_feat, target_bbox, learning_rate)

        # Decide the number of iterations to run
        num_iters = 0
        if hard_neg_flag:
            num_iters = self.optimizer_cfg.get('hard_neg_iters', None)
        elif (frame_num - 1) % self.update_cfg['train_skipping'] == 0:
            num_iters = self.optimizer_cfg.get('update_iters', None)

        if num_iters > 0:
            # Get inputs for the DiMP filter optimizer module
            samples = self.memo.training_samples[:self.memo.num_samples, ...]
            target_bboxes = self.memo.bboxes[:self.memo.num_samples, :].clone()
            sample_weights = self.memo.sample_weights[:self.memo.num_samples]

            # Run the filter optimizer module
            with torch.no_grad():
                self.target_filter = self.filter_optimizer(
                    self.target_filter,
                    num_iters=num_iters,
                    feat=samples,
                    bboxes=target_bboxes,
                    sample_weights=sample_weights)
