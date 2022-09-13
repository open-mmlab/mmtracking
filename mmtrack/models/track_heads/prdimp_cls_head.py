# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from addict import Dict
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from mmengine.model import BaseModule
from torch import Tensor, nn

from mmtrack.registry import MODELS
from mmtrack.utils import OptConfigType, SampleList, max_last2d
from ..task_modules.filter import filter as filter_layer


@MODELS.register_module()
class PrDiMPClsHead(BaseModule):
    """PrDiMP classification head.

    Args:
        in_dim (int, optional): The dim of input feature. Defaults to 1024.
        out_dim (int, optional): The dim of output. Defaults to 512.
        filter_initializer (dict, optional): The configuration of filter
            initializer. Defaults to None.
        filter_optimizer (dict, optional): The configuration of filter
            optimizer. Defaults to None.
        locate_cfg (dict, optional): The configuration of bbox location.
            Defaults to None.
        update_cfg (dict, optional): The configuration of updating tracking
            state in memory. Defaults to None.
        optimizer_cfg (dict, optional): The configuration of optimizer.
            Defaults to None.
        loss_cls (dict, optional): The configuration of classification
            loss. Defaults to None.
        train_cfg (dict, optional): The configuration of training.
            Defaults to None.
    """

    def __init__(self,
                 in_dim: int = 1024,
                 out_dim: int = 512,
                 filter_initializer: OptConfigType = None,
                 filter_optimizer: OptConfigType = None,
                 locate_cfg: OptConfigType = None,
                 update_cfg: OptConfigType = None,
                 optimizer_cfg: OptConfigType = None,
                 loss_cls: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 **kwargs):
        super().__init__()
        filter_size = filter_initializer['filter_size']
        self.filter_initializer = MODELS.build(filter_initializer)
        self.filter_optimizer = MODELS.build(filter_optimizer)
        self.feat_norm_scale = math.sqrt(1.0 /
                                         (out_dim * filter_size * filter_size))
        self.channel_mapping = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, bias=False))

        self.locate_cfg = locate_cfg
        self.update_cfg = update_cfg
        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg

        self.loss_cls = MODELS.build(loss_cls)

        if isinstance(filter_size, (int, float)):
            filter_size = [filter_size, filter_size]
        self.filter_size = torch.tensor(filter_size, dtype=torch.float32)
        self.feat_size = torch.tensor(
            train_cfg['feat_size'], dtype=torch.float32)
        self.img_size = torch.tensor(
            train_cfg['img_size'], dtype=torch.float32)

    def init_weights(self):
        """Initialize the parameters of this module."""
        for m in self.channel_mapping:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.filter_initializer.init_weights()

    def get_cls_feats(self, backbone_feats: Tensor) -> Tensor:
        """Get features for classification.

        Args:
            backbone_feats (Tensor): The features from backbone.

        Returns:
            Tensor: The features for classification.
        """
        cls_feats = self.channel_mapping(backbone_feats)
        scale_factor = (torch.tensor(cls_feats.shape[1:]).prod() / (torch.sum(
            (cls_feats**2).reshape(cls_feats.shape[0], 1, 1, -1),
            dim=3,
            keepdim=True) + 1e-5)).sqrt()
        cls_feats = cls_feats * self.feat_norm_scale * scale_factor

        return cls_feats

    def init_classifier(self,
                        backbone_feats: Tensor,
                        target_bboxes: Tensor,
                        dropout_probs: Optional[List] = None):
        """Initialize the filter and memory in the classifier.

        Args:
            backbone_feats (Tensor): The features from backbone.
            target_bboxes (Tensor): in [cx, cy, w, h] format.
            dropout_probs (list, optional): Defaults to None.
        """
        cls_feats = self.get_cls_feats(backbone_feats)

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
        init_filter = self.filter_initializer(cls_feats, target_bboxes_xyxy)
        self.target_filter = self.filter_optimizer(
            init_filter,
            feat=cls_feats,
            bboxes=target_bboxes,
            num_iters=self.optimizer_cfg['init_update_iters'])

        # Initialize memory
        self.init_memory(cls_feats, target_bboxes)

    def init_memory(self, aug_feats: Tensor, target_bboxes: Tensor):
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

    def forward(self, backbone_feats: Tensor) -> Tuple[Tensor, Tensor]:
        """Run classifier on the backbone features.

        Args:
            backbone_feats (Tensor): the features from the last layer of
                backbone

        Returns:
            scores (Tensor): of shape (bs, 1, h, w)
            feats (Tensor): features for classification.
        """
        feats = self.get_cls_feats(backbone_feats)
        scores = filter_layer.apply_filter(feats, self.target_filter)
        return scores, feats

    def update_memory(self,
                      target_bbox: Tensor,
                      learning_rate: Optional[float] = None):
        """Update the tracking state in memory.

        Args:
            target_bbox (Tensor): of shape (1,4) in [x, y, w, h] format.
            learning_rate (float, optional): The learning rate about updating.
                Defaults to None.
        """
        # Update weights and get replace ind
        replace_ind = self.update_sample_weights(learning_rate)
        self.memo.replace_ind = replace_ind

        # Update training samples and bboxes in memory
        self.memo.training_samples[replace_ind:replace_ind + 1,
                                   ...] = self.memo.sample_feat
        self.memo.bboxes[replace_ind, :] = target_bbox
        self.memo.num_samples += 1

    def update_sample_weights(self, learning_rate=None) -> int:
        """Update the weights of samples in memory.

        Args:
            learning_rate (int, optional): The learning rate of updating
                samples in memory. Defaults to None.

        Returns:
            (int): the index of updated samples in memory.
        """

        init_sample_weight = self.update_cfg['init_samples_min_weight']
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
                          target_bbox: Tensor,
                          frame_num: int,
                          hard_neg_flag: Optional[bool] = False):
        """Update the classifier with the refined bbox.

        Args:
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
        if hard_neg_flag:
            self.update_memory(target_bbox, learning_rate)

        # Decide the number of iterations to run
        num_iters = 0
        if hard_neg_flag:
            num_iters = self.optimizer_cfg['hard_neg_iters']
        elif (frame_num - 1) % self.update_cfg['train_skipping'] == 0:
            num_iters = self.optimizer_cfg['update_iters']

        if num_iters > 0:
            # Get inputs for the DiMP filter optimizer module
            samples = self.memo.training_samples[:self.memo.num_samples, ...]
            target_bboxes = self.memo.bboxes[:self.memo.num_samples, :].clone()
            sample_weights = self.memo.sample_weights[:self.memo.num_samples]

            # Run the filter optimizer module
            self.target_filter = self.filter_optimizer(
                self.target_filter,
                num_iters=num_iters,
                feat=samples,
                bboxes=target_bboxes,
                sample_weights=sample_weights)

    def predict(self, backbone_feats: Tuple[Tensor], data_samples: SampleList,
                prev_bbox: Tensor, sample_center: Tensor,
                scale_factor: float) -> Tuple[Tensor, Tensor, bool]:
        """Perform forward propagation of the tracking head and predict
        tracking results on the features of the upstream network.

        Args:
            backbone_feats (Tuple[Tensor]): The features from backbone.
            data_samples (List[:obj:`TrackDataSample`]): The Data
                Samples. It usually includes information such as `gt_instance`.
            sample_center (Tensor): The coordinate of the sample center based
                on the original image.
            prev_bbox (Tensor): of shape (4, ) in [cx, cy, w, h] format.
            scale_factor (float): scale factor.

        Returns:
            new_bbox_center (Tensor): The center of the new bbox.
            scores_map (Tensor): The score map from the classifier.
            state (Tensor): The tracking state.
        """
        # ``self.memo.sample_feat`` is used to update the training samples
        # in the memory on some conditions.
        scores_raw, self.memo.sample_feat = self(backbone_feats[-1])
        scores_map = torch.softmax(
            scores_raw.view(-1), dim=0).view(scores_raw.shape)

        new_bbox_center, state = self.predict_by_feat(scores_map, prev_bbox,
                                                      sample_center,
                                                      scale_factor)

        return new_bbox_center, scores_map, state

    def _gen_2d_hanning_windows(self,
                                size: Sequence,
                                device: str = 'cuda') -> Tensor:
        """Generate 2D hanning window.

        Args:
            size (Sequence): The size of 2d hanning window.
            device (str): Device the tensor will be put on. Defaults to 'cuda'.

        Returns:
            Tensor: 2D hanning window with shape
            (num_base_anchors[i] * featmap_sizes[i][0] * featmap_sizes[i][1]).
        """

        def hanning_1d(s):
            return 0.5 * (1 - torch.cos(
                (2 * math.pi / (s + 1)) * torch.arange(1, s + 1).float()))

        hanning_win = hanning_1d(size[0]).reshape(-1, 1) * hanning_1d(
            size[1]).reshape(1, -1)

        return hanning_win.to(device)

    def predict_by_feat(self, scores: Tensor, prev_bbox: Tensor,
                        sample_center: Tensor,
                        scale_factor: float) -> Tuple[Tensor, bool]:
        """Track `prev_bbox` to current frame based on the output of network.

        Args:
            scores (Tensor): It's of shape (1, h, w) or (h, w).
            prev_bbox (Tensor): It's of shape (4,) in [cx, cy, w, h] format.
            sample_center (Tensor): The center of the cropped
                sample on the original image. It's of shape (1,2) or (2,) in
                [x, y] format.
            scale_factor (float): The scale of the cropped sample.
                It's of shape (1,) when it's a tensor.

        Return:
            Tensor: The displacement of the target to the center of original
                image
            bool: The tracking state.
        """
        sample_center = sample_center.squeeze()
        scores = scores.squeeze()
        prev_bbox = prev_bbox.squeeze()
        assert scores.dim() == 2
        assert sample_center.dim() == prev_bbox.dim() == 1

        score_size = torch.tensor([scores.shape[-1], scores.shape[-2]])
        output_size = (score_size - (self.filter_size + 1) % 2).to(
            scores.device)
        score_center = (score_size / 2).to(scores.device)

        scores_hn = scores
        if self.locate_cfg.get('hanning_window', False):
            scores_hn = scores * self._gen_2d_hanning_windows(
                score_size, device=scores.device)

        max_score, max_pos = max_last2d(scores_hn)
        max_pos = max_pos.flip(0).float()
        # the displacement of target to the center of score map
        target_disp_score_map = max_pos - score_center
        # the ratio of the size of original image to to that of score map
        ratio_size = (self.test_cfg['img_sample_size'] /
                      output_size) * scale_factor
        # the displcement of the target to the center of original image
        target_disp = target_disp_score_map * ratio_size

        # Handle different cases
        # 1. Target is not found
        if max_score.item() < self.locate_cfg['no_target_min_score']:
            return target_disp + sample_center, 'not_found'

        # Analysis whether there is a distractor
        # Calculate the size of neighborhood near the current target
        target_neigh_sz = self.locate_cfg['target_neighborhood_scale'] * (
            prev_bbox[2:4] / ratio_size)

        top_left = (max_pos - target_neigh_sz / 2).round().long()
        top_left = torch.clamp_min(top_left, 0).tolist()
        bottom_right = (max_pos + target_neigh_sz / 2).round().long()
        bottom_right = torch.clamp_max(bottom_right,
                                       score_size.min().item()).tolist()
        scores_masked = scores.clone()
        scores_masked[top_left[1]:bottom_right[1],
                      top_left[0]:bottom_right[0]] = 0

        # Find new maximum except the neighborhood of the target
        second_max_score, second_max_pos = max_last2d(scores_masked)
        second_max_pos = second_max_pos.flip(0).float().view(-1)
        distractor_disp_score_map = second_max_pos - score_center
        distractor_disp = distractor_disp_score_map * ratio_size
        # The displacement of previout target bbox to the center of the score
        # map.
        # Note that `sample_center`` may not be equal to the center of previous
        # tracking bbox due to different cropping mode
        prev_target_disp_score_map = (prev_bbox[:2] -
                                      sample_center) / ratio_size

        # 2. There is a distractor
        if second_max_score > self.locate_cfg['distractor_thres'] * max_score:
            target_disp_diff = torch.sqrt(
                torch.sum(
                    (target_disp_score_map - prev_target_disp_score_map)**2))
            # `distractor_disp_diff` is the displacement between current
            # tracking bbox and previous tracking bbox.
            distractor_disp_diff = torch.sqrt(
                torch.sum((distractor_disp_score_map -
                           prev_target_disp_score_map)**2))
            disp_diff_thres = self.locate_cfg[
                'dispalcement_scale'] * score_size.prod().float().sqrt() / 2

            if (distractor_disp_diff > disp_diff_thres
                    and target_disp_diff < disp_diff_thres):
                return target_disp + sample_center, 'hard_negative'
            if (distractor_disp_diff < disp_diff_thres
                    and target_disp_diff > disp_diff_thres):
                # The true target may be the `distractor` instead of the
                # `target` on this frame
                return distractor_disp + sample_center, 'hard_negative'
            else:
                # If both the displacement of target and distractor is larger
                # or smaller than the threshold, return the displacement of the
                # highest score.
                return target_disp + sample_center, 'uncertain'

        # 3. There is a hard negative object
        if (second_max_score > self.locate_cfg['hard_neg_thres'] * max_score
                and second_max_score > self.locate_cfg['no_target_min_score']):
            return target_disp + sample_center, 'hard_negative'

        # 4. Normal target
        return target_disp + sample_center, 'normal'

    def _gauss_1d(self,
                  size: Tensor,
                  sigma: float,
                  center: Tensor,
                  end_pad: int = 0,
                  return_density: bool = True):
        """Generate gauss labels.

        Args:
            size (Tensor): The size of score map with (2, ) shape.
            sigma (float): Standard deviations.
            center (Tensor): of (N, ) shape.
            end_pad (int, optional): The padding size.. Defaults to 0.
            return_density (bool, optional): Whether to return density.
                Defaults to True.

        Returns:
            Tensor: Gauss labels with (N, score_map_size) shape.
        """
        k = torch.arange(-(size - 1) / 2, (size + 1) / 2 + end_pad).reshape(
            1, -1).to(center.device)
        gauss = torch.exp(-1.0 / (2 * sigma**2) *
                          (k - center.reshape(-1, 1))**2)
        if not return_density:
            return gauss
        else:
            return gauss / (math.sqrt(2 * math.pi) * sigma)

    def _gauss_2d(self,
                  size: Tensor,
                  sigma: float,
                  center: Tensor,
                  end_pad: Union[Tensor, List, Tuple] = (0, 0),
                  return_density: bool = True):
        """Generate gauss labels.

        Args:
            size (Tensor): The size of score map with (2, ) shape.
            sigma (Tensor): Standard deviations.
            center (Tensor): The center of bbox with (N, 2) shape.
            end_pad (Union[Tensor, List, Tuple], optional): The padding size.
                Defaults to (0, 0).
            return_density (bool, optional): Whether to return density.
                Defaults to True.

        Returns:
            Tensor: Gauss labels with (N, score_map_size, score_map_size)
                shape.
        """
        if isinstance(sigma, (float, int)):
            sigma = (sigma, sigma)

        gauss_1d_out1 = self._gauss_1d(
            size[0].item(),
            sigma[0],
            center[:, 0],
            end_pad[0],
            return_density=return_density)
        gauss_1d_out1 = gauss_1d_out1.reshape(center.shape[0], 1, -1)

        gauss_1d_out2 = self._gauss_1d(
            size[1].item(),
            sigma[1],
            center[:, 1],
            end_pad[1],
            return_density=return_density)
        gauss_1d_out2 = gauss_1d_out2.reshape(center.shape[0], -1, 1)

        return gauss_1d_out1 * gauss_1d_out2

    def get_targets(self, target_center: Tensor) -> Tensor:
        """Generate the training targets for search images.

        Args:
            target_center (Tensor): The center of target bboxes with (N, 2)
                shape, in [x, y] format.

        Returns:
            Tensor: The distribution of gauss labels with
                (N, score_map_size, score_map_size) shape.
        """
        # TODO simplify the different devices
        img_size = self.img_size.to(target_center.device)
        filter_size = self.filter_size.to(target_center.device)
        feat_size = self.feat_size.to(target_center.device)

        # set the center of image as coordinate origin
        target_center_norm = (target_center - img_size / 2.) / img_size

        center = feat_size * target_center_norm + 0.5 * torch.fmod(
            filter_size + 1, 2)

        sigma = self.train_cfg['sigma_factor'] * feat_size.prod().sqrt().item()

        if self.train_cfg['end_pad_if_even']:
            end_pad = (torch.fmod(filter_size, 2) == 0).to(filter_size)
        else:
            end_pad = torch.zeros(2).to(filter_size)

        # generate gauss labels and density
        # Both of them are of shape (N, feat_size, feat_size)
        gauss_label_density = self._gauss_2d(
            feat_size, sigma, center, end_pad,
            self.train_cfg['use_gauss_density'])
        # continue to process label density
        feat_area = (feat_size + end_pad).prod()
        gauss_label_density = (1.0 - self.train_cfg['gauss_label_bias']
                               ) * gauss_label_density + self.train_cfg[
                                   'gauss_label_bias'] / feat_area

        mask = (gauss_label_density > self.train_cfg['label_density_threshold']
                ).to(gauss_label_density)
        gauss_label_density *= mask

        if self.train_cfg['label_density_norm']:
            g_sum = gauss_label_density.sum(dim=(-2, -1))
            valid = g_sum > 0.01
            gauss_label_density[valid, :, :] /= g_sum[valid].reshape(-1, 1, 1)
            gauss_label_density[~valid, :, :] = 1.0 / (
                gauss_label_density.shape[-2] * gauss_label_density.shape[-1])
        gauss_label_density *= 1.0 - self.train_cfg['label_density_shrink']

        return gauss_label_density

    def loss(self, template_feats: Tuple[Tensor], search_feats: Tuple[Tensor],
             batch_data_samples: SampleList, **kwargs) -> dict:
        """Perform forward propagation and loss calculation of the tracking
        head on the features of the upstream network.

        Args:
            template_feats (tuple[Tensor, ...]): Tuple of Tensor with
                shape (N, C, H, W) denoting the multi level feature maps of
                exemplar images.
            search_feats (tuple[Tensor, ...]): Tuple of Tensor with shape
                (N, C, H, W) denoting the multi level feature maps of
                search images.
            batch_data_samples (List[:obj:`TrackDataSample`]): The Data
                Samples. It usually includes information such as `gt_instance`.

        Returns:
            dict: A dictionary of loss components.
        """
        batch_gt_bboxes = []
        batch_img_metas = []
        batch_search_gt_bboxes = []

        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_bboxes.append(data_sample.gt_instances['bboxes'])
            batch_search_gt_bboxes.append(
                data_sample.search_gt_instances['bboxes'])

        num_imgs_per_seq = batch_data_samples[0].gt_instances['bboxes'].shape[
            0]
        num_search_imgs_per_seq = batch_data_samples[0].search_gt_instances[
            'bboxes'].shape[0]

        # Extract features
        template_feats = self.get_cls_feats(template_feats[-1])
        template_feats = template_feats.reshape(num_imgs_per_seq, -1,
                                                *template_feats.shape[-3:])
        search_feats = self.get_cls_feats(search_feats[-1])
        search_feats = search_feats.reshape(num_search_imgs_per_seq, -1,
                                            *search_feats.shape[-3:])

        return self.loss_by_feat(template_feats, search_feats, batch_gt_bboxes,
                                 batch_search_gt_bboxes)

    def loss_by_feat(self, template_feats: Tensor, search_feats: Tensor,
                     batch_gt_bboxes: Tensor,
                     batch_search_gt_bboxes: Tensor) -> dict:
        """Compute loss.

        Args:
            modulations (Tuple[Tensor]): The modulation features.
            iou_feats (Tuple[Tensor]): The features for iou prediction.
            batch_gt_bboxes (Tensor): The gt_bboxes in a batch.
            batch_search_gt_bboxes (Tensor): The search gt_bboxes in a batch.

        Returns:
            dict: a dictionary of loss components.
        """
        # Train filter
        batch_gt_bboxes = torch.stack(batch_gt_bboxes, dim=1)
        # filter_iter is a list, and each of them is of shape
        # (bs, C, self.filter_size, self.filter_size)
        init_filter = self.filter_initializer(template_feats,
                                              batch_gt_bboxes.view(-1, 4))
        bboxes_cxcywh = bbox_xyxy_to_cxcywh(batch_gt_bboxes)
        _, filters_all_iters, _ = self.filter_optimizer(
            init_filter, feat=template_feats, bboxes=bboxes_cxcywh)

        # Classify samples using all filters
        # each item in ``target_scores`` is of shape
        # (num_search_imgs_per_seq, bs, score_map_size, score_map_size)
        target_scores = [
            filter_layer.apply_filter(search_feats, filter_weight)
            for filter_weight in filters_all_iters
        ]

        batch_search_gt_bboxes = torch.stack(batch_search_gt_bboxes, dim=1)
        search_gt_bboxes = batch_search_gt_bboxes.view(-1, 4)
        target_center = (search_gt_bboxes[:, :2] +
                         search_gt_bboxes[:, 2:4]) / 2.
        prob_labels_density = self.get_targets(target_center)
        # of shape (num_search_imgs_per_seq, bs, score_map_size, score_map_size) # noqa: E501
        prob_labels_density = prob_labels_density.view(
            batch_search_gt_bboxes.shape[0], -1,
            *prob_labels_density.shape[-2:])

        # compute loss
        loss_cls_list = [
            self.loss_cls(score, prob_labels_density, grid_dim=(-2, -1))
            for score in target_scores
        ]
        loss_cls_final = self.train_cfg['loss_weights'][
            'cls_final'] * loss_cls_list[-1]
        loss_cls_init = self.train_cfg['loss_weights'][
            'cls_init'] * loss_cls_list[0]

        if isinstance(self.train_cfg['loss_weights']['cls_iter'], list):
            loss_cls_iters = sum([
                weight * loss for weight, loss in zip(
                    self.train_cfg['loss_weights']['cls_iter'],
                    loss_cls_list[1:-1])
            ])
        else:
            loss_cls_iters = (self.train_cfg['loss_weights']['cls_iter'] /
                              (len(loss_cls_list) - 2)) * sum(
                                  loss_cls_list[1:-1])
        losses = dict(
            loss_cls_init=loss_cls_init,
            loss_cls_iter=loss_cls_iters,
            loss_cls_final=loss_cls_final)

        return losses
