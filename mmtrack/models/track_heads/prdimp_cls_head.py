# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn.functional as F
from mmdet.models import HEADS
from mmdet.models.builder import build_head, build_loss
from torch import nn

from mmtrack.core.bbox import bbox_xyxy_to_x1y1wh
from mmtrack.core.filter import filter as filter_layer
from mmtrack.core.utils import TensorList


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


def softmax_reg(x: torch.Tensor, dim, reg=None):
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
        self.loss_cls = build_loss(loss_cls)
        self.loss_weights = train_cfg['loss_weights']

        self.locate_cfg = locate_cfg
        self.update_cfg = update_cfg
        self.optimizer_cfg = optimizer_cfg

        if isinstance(filter_size, (int, float)):
            filter_size = [filter_size, filter_size]
        self.filter_size = torch.tensor(filter_size, dtype=torch.float32)
        self.feat_size = torch.tensor(
            train_cfg['feat_size'], dtype=torch.float32)
        self.img_size = torch.tensor(
            train_cfg['img_size'], dtype=torch.float32)
        self.sigma_factor = train_cfg['sigma_factor']
        self.end_pad_if_even = train_cfg['end_pad_if_even']

        self.gauss_label_bias = train_cfg['gauss_label_bias']
        self.use_gauss_density = train_cfg['use_gauss_density']
        self.label_density_norm = train_cfg['label_density_norm']
        self.label_density_threshold = train_cfg['label_density_threshold']
        self.label_density_shrink = train_cfg['label_density_shrink']

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

    def forward(self, train_feat, test_feat, gt_bboxes, search_gt_bboxes,
                *args, **kwargs):
        """Learns a target classification filter based on the train samples and
            return the resulting classification scores on the test samples.
            The forward function is ONLY used for training. Call the individual
            functions during tracking.
        args:
            train_feat (Tensor):  of (num_template_seq * bs, C , H, W) shape
            test_feat (Tensor):  of (num_search_seq*bs, C , H, W) shape
            gt_bboxes (Tensor): ground truth bboxes for search images
                with shape (num_template_seq, bs, 5) in
                [0., tl_x, tl_y, br_x, br_y] format.
            search_gt_bboxes (list[Tensor]): ground truth bboxes for search
                images with shape (num_search_seq, bs, 5) in
                [0., tl_x, tl_y, br_x, br_y] format.
        """
        train_num_seq = gt_bboxes.shape[0]
        test_num_seq = search_gt_bboxes.shape[0]

        # Extract features
        train_feat = self.cls_feature_extractor(train_feat)
        train_feat = train_feat.reshape(train_num_seq, -1,
                                        *train_feat.shape[-3:])
        test_feat = self.cls_feature_extractor(test_feat)
        test_feat = test_feat.reshape(test_num_seq, -1, *test_feat.shape[-3:])

        # Train filter
        # convert the bbox from [x1, y1, x2, y2] to [x1, y1, w, h]
        bboxes_xywh = bbox_xyxy_to_x1y1wh(gt_bboxes.view(-1, 5)[:, 1:])
        bboxes_xywh = bboxes_xywh.view(train_num_seq, -1, 4)
        # filter_iter is a list, and each of them is of shape
        # (bs, C, self.filter_size, self.filter_size)
        _, filter_iter, _ = self.get_filter(train_feat, bboxes_xywh, *args,
                                            **kwargs)

        # Classify samples using all filters
        # each of target_scores is of shape
        # (num_search_seq, bs, score_map_size, score_map_size)
        target_scores = [self.classify(test_feat, f) for f in filter_iter]

        search_gt_bboxes = search_gt_bboxes.view(-1,
                                                 *search_gt_bboxes.shape[2:])
        target_center = (search_gt_bboxes[:, 1:3] +
                         search_gt_bboxes[:, 3:5]) / 2.
        # of shape (num_template_seq*bs, score_map_size, score_map_size)
        prob_labels_density = self.get_targets(target_center)
        prob_labels_density = prob_labels_density.view(
            test_num_seq, -1, *prob_labels_density.shape[-2:])

        # compute loss
        loss_cls_list = [
            self.loss_cls(score, prob_labels_density, grid_dim=(-2, -1))
            for score in target_scores
        ]
        loss_cls_final = self.loss_weights['cls_final'] * loss_cls_list[-1]
        loss_cls_init = self.loss_weights['cls_init'] * loss_cls_list[0]

        if isinstance(self.loss_weights['cls_iter'], list):
            loss_cls_iter = sum([
                a * b for a, b in zip(self.loss_weights['cls_iter'],
                                      loss_cls_list[1:-1])
            ])
        else:
            loss_cls_iter = (self.loss_weights['cls_iter'] /
                             (len(loss_cls_list) - 2)) * sum(
                                 loss_cls_list[1:-1])
        losses = dict(
            loss_cls_init=loss_cls_init,
            loss_cls_iter=loss_cls_iter,
            loss_cls_final=loss_cls_final)

        return losses

    def _gauss_1d(self, sz, sigma, center, end_pad=0, return_density=True):
        k = torch.arange(-(sz - 1) / 2,
                         (sz + 1) / 2 + end_pad).reshape(1,
                                                         -1).to(center.device)
        gauss = torch.exp(-1.0 / (2 * sigma**2) *
                          (k - center.reshape(-1, 1))**2)
        if not return_density:
            return gauss
        else:
            return gauss / (math.sqrt(2 * math.pi) * sigma)

    def _gauss_2d(self,
                  sz,
                  sigma,
                  center,
                  end_pad=(0, 0),
                  return_density=True):
        if isinstance(sigma, (float, int)):
            sigma = (sigma, sigma)

        gauss_1d_out1 = self._gauss_1d(
            sz[0].item(),
            sigma[0],
            center[:, 0],
            end_pad[0],
            return_density=return_density)
        gauss_1d_out1 = gauss_1d_out1.reshape(center.shape[0], 1, -1)

        gauss_1d_out2 = self._gauss_1d(
            sz[1].item(),
            sigma[1],
            center[:, 1],
            end_pad[1],
            return_density=return_density)
        gauss_1d_out2 = gauss_1d_out2.reshape(center.shape[0], -1, 1)

        return gauss_1d_out1 * gauss_1d_out2

    def get_targets(self, target_center):
        """_summary_

        Args:
            target_center (Tensor): [N, 2] in [cx, cy] format
        """
        self.img_size = self.img_size.to(target_center.device)
        self.filter_size = self.filter_size.to(target_center.device)
        self.feat_size = self.feat_size.to(target_center.device)

        # set the center of image as coordinate origin
        target_center_norm = (target_center -
                              self.img_size / 2.) / self.img_size

        center = self.feat_size * target_center_norm + 0.5 * torch.fmod(
            self.filter_size + 1, 2)

        sigma = self.sigma_factor * self.feat_size.prod().sqrt().item()

        if self.end_pad_if_even:
            end_pad = (torch.fmod(self.filter_size,
                                  2) == 0).to(self.filter_size)
        else:
            end_pad = torch.zeros(2).to(self.filter_size)

        # generate gauss labels and density
        # Both of them are of shape (N, self.feat_size, self.feat_size)
        gauss_label_density = self._gauss_2d(self.feat_size, sigma, center,
                                             end_pad, self.use_gauss_density)
        # continue to process label density
        feat_area = (self.feat_size + end_pad).prod()
        gauss_label_density = (
            1.0 - self.gauss_label_bias
        ) * gauss_label_density + self.gauss_label_bias / feat_area

        mask = (gauss_label_density >
                self.label_density_threshold).to(gauss_label_density)
        gauss_label_density *= mask

        if self.label_density_norm:
            g_sum = gauss_label_density.sum(dim=(-2, -1))
            valid = g_sum > 0.01
            gauss_label_density[valid, :, :] /= g_sum[valid].reshape(-1, 1, 1)
            gauss_label_density[~valid, :, :] = 1.0 / (
                gauss_label_density.shape[-2] * gauss_label_density.shape[-1])
        gauss_label_density *= 1.0 - self.label_density_shrink

        return gauss_label_density

    def init_classifier(self, backbone_feats, target_bboxes, dropout_cfg=None):

        self.target_boxes = target_bboxes.new_zeros(
            self.update_cfg['sample_memory_size'], 4)
        self.target_boxes[:target_bboxes.shape[0], :] = target_bboxes

        # Initialize classifier, using the `layer3` features
        with torch.no_grad():
            cls_feat = self.cls_feature_extractor(backbone_feats)
        if dropout_cfg is not None:
            num, prob = dropout_cfg
            # from mmdet.apis import set_random_seed
            # set_random_seed(1)
            aug_feat = F.dropout2d(
                cls_feat[0:1, ...].expand(num, -1, -1, -1),
                p=prob,
                training=True)
            cls_feat = torch.cat([cls_feat, aug_feat])

        # Get target filter by running the discriminative model prediction
        # module
        with torch.no_grad():
            self.target_filter, _, _ = self.get_filter(
                cls_feat,
                target_bboxes,
                num_iter=self.optimizer_cfg['net_opt_iter'])

        return cls_feat

    def localize_target(self, scores, sample_pos, sample_scales, sample_size,
                        prev_bbox_hw, prev_bbox_yx):
        """Run the target localization."""

        scores = scores.squeeze(1)

        preprocess_method = self.locate_cfg.get('score_preprocess', 'none')
        if preprocess_method == 'none':
            pass
        elif preprocess_method == 'exp':
            scores = scores.exp()
        elif preprocess_method == 'softmax':
            reg_val = getattr(self.filter_optimizer, 'softmax_reg', None)
            scores_view = scores.view(scores.shape[0], -1)
            scores_softmax = softmax_reg(scores_view, dim=-1, reg=reg_val)
            scores = scores_softmax.view(scores.shape)
        else:
            raise Exception('Unknown score_preprocess in params.')

        return self.localize_advanced(scores, sample_pos, sample_scales,
                                      sample_size, prev_bbox_hw, prev_bbox_yx)

    def localize_advanced(self, scores, sample_pos, sample_scales, sample_size,
                          prev_bbox_hw, prev_bbox_yx):
        """Run the target advanced localization (as in ATOM).

        sample_scale: target_scale is the ratio of the size of cropped image
            to the size of resized image.
        sample_size: the size of input size
        """

        sz = scores.shape[-2:]
        score_sz = torch.tensor(list(sz))
        output_sz = score_sz - (self.filter_size + 1) % 2
        score_center = (score_sz - 1) / 2

        max_score1, max_disp1 = max2d(scores)
        _, scale_ind = torch.max(max_score1, dim=0)
        sample_scale = sample_scales[scale_ind]
        max_score1 = max_score1[scale_ind]
        max_disp1 = max_disp1[scale_ind, ...].float().cpu().view(-1)
        target_disp1 = max_disp1 - score_center
        translation_vec1 = target_disp1 * (sample_size /
                                           output_sz) * sample_scale

        if max_score1.item() < self.locate_cfg['target_not_found_threshold']:
            return translation_vec1, scale_ind, scores, 'not_found'
        if max_score1.item() < self.locate_cfg.get('uncertain_threshold',
                                                   -float('inf')):
            return translation_vec1, scale_ind, scores, 'uncertain'
        if max_score1.item() < self.locate_cfg.get('hard_sample_threshold',
                                                   -float('inf')):
            return translation_vec1, scale_ind, scores, 'hard_negative'

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
        scores_masked = scores[scale_ind:scale_ind + 1, ...].clone()
        scores_masked[..., tneigh_top:tneigh_bottom,
                      tneigh_left:tneigh_right] = 0

        # Find new maximum
        max_score2, max_disp2 = max2d(scores_masked)
        max_disp2 = max_disp2.float().cpu().view(-1)
        target_disp2 = max_disp2 - score_center
        translation_vec2 = target_disp2 * (sample_size /
                                           output_sz) * sample_scale

        prev_target_vec = (prev_bbox_yx - sample_pos[scale_ind, :]) / (
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
                return translation_vec1, scale_ind, scores, 'hard_negative'
            if disp_norm2 < disp_threshold and disp_norm1 > disp_threshold:
                return translation_vec2, scale_ind, scores, 'hard_negative'
            if disp_norm2 > disp_threshold and disp_norm1 > disp_threshold:
                return translation_vec1, scale_ind, scores, 'uncertain'

            # If also the distractor is close, return with highest score
            return translation_vec1, scale_ind, scores, 'uncertain'

        if (max_score2 >
                self.locate_cfg['hard_negative_threshold'] * max_score1 and
                max_score2 > self.locate_cfg['target_not_found_threshold']):
            return translation_vec1, scale_ind, scores, 'hard_negative'

        return translation_vec1, scale_ind, scores, 'normal'

    def init_memory(self, train_x: TensorList):
        # Initialize first-frame spatial training samples
        self.num_init_samples = train_x.size(0)
        init_sample_weights = TensorList(
            [x.new_ones(1) / x.shape[0] for x in train_x])

        # Sample counters and weights for spatial
        self.num_stored_samples = self.num_init_samples.copy()
        self.previous_replace_ind = [None] * len(self.num_stored_samples)
        self.sample_weights = TensorList([
            x.new_zeros(self.update_cfg['sample_memory_size']) for x in train_x
        ])
        for sw, init_sw, num in zip(self.sample_weights, init_sample_weights,
                                    self.num_init_samples):
            sw[:num] = init_sw

        # Initialize memory
        self.training_samples = TensorList([
            x.new_zeros(self.update_cfg['sample_memory_size'], x.shape[1],
                        x.shape[2], x.shape[3]) for x in train_x
        ])

        for ts, x in zip(self.training_samples, train_x):
            ts[:x.shape[0], ...] = x

    def update_memory(self,
                      sample_x: TensorList,
                      target_box,
                      learning_rate=None):
        # Update weights and get replace ind
        replace_ind = self.update_sample_weights(self.sample_weights,
                                                 self.previous_replace_ind,
                                                 self.num_stored_samples,
                                                 self.num_init_samples,
                                                 learning_rate)
        self.previous_replace_ind = replace_ind

        # Update sample and label memory
        for train_samp, x, ind in zip(self.training_samples, sample_x,
                                      replace_ind):
            train_samp[ind:ind + 1, ...] = x

        # Update bb memory
        self.target_boxes[replace_ind[0], :] = target_box

        self.num_stored_samples += 1

    def update_sample_weights(self,
                              sample_weights,
                              previous_replace_ind,
                              num_stored_samples,
                              num_init_samples,
                              learning_rate=None):
        # Update weights and get index to replace
        replace_ind = []
        for sw, prev_ind, num_samp, num_init in zip(sample_weights,
                                                    previous_replace_ind,
                                                    num_stored_samples,
                                                    num_init_samples):
            lr = learning_rate
            if lr is None:
                lr = self.locate_cfg.learning_rate

            init_samp_weight = self.locate_cfg.get(
                'init_samples_minimum_weight', None)
            if init_samp_weight == 0:
                init_samp_weight = None
            s_ind = 0 if init_samp_weight is None else num_init

            if num_samp == 0 or lr == 1:
                sw[:] = 0
                sw[0] = 1
                r_ind = 0
            else:
                # Get index to replace
                if num_samp < sw.shape[0]:
                    r_ind = num_samp
                else:
                    _, r_ind = torch.min(sw[s_ind:], 0)
                    r_ind = r_ind.item() + s_ind

                # Update weights
                if prev_ind is None:
                    sw /= 1 - lr
                    sw[r_ind] = lr
                else:
                    sw[r_ind] = sw[prev_ind] / (1 - lr)

            sw /= sw.sum()
            if init_samp_weight is not None and sw[:num_init].sum(
            ) < init_samp_weight:
                sw /= init_samp_weight + sw[num_init:].sum()
                sw[:num_init] = init_samp_weight / num_init

            replace_ind.append(r_ind)

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
            self.update_memory(
                TensorList([train_x]), target_box, learning_rate)

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
            samples = self.training_samples[0][:self.num_stored_samples[0],
                                               ...]
            target_boxes = self.target_boxes[:self.
                                             num_stored_samples[0], :].clone()
            sample_weights = self.sample_weights[0][:self.
                                                    num_stored_samples[0]]

            # Run the filter optimizer module
            with torch.no_grad():
                self.target_filter, _, losses = self.filter_optimizer(
                    self.target_filter,
                    num_iter=num_iter,
                    feat=samples,
                    bb=target_boxes,
                    sample_weight=sample_weights)

    def classify(self, feat, filter=None):
        """Run classifier (filter) on the features (feat)."""
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
