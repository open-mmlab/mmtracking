# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from mmcv.cnn.bricks import ConvModule
from mmdet.structures.bbox.transforms import (bbox_cxcywh_to_xyxy,
                                              bbox_xyxy_to_cxcywh)
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from torch import Tensor

from mmtrack.registry import MODELS, TASK_UTILS
from mmtrack.utils import InstanceList, SampleList
from ..task_modules.track import depthwise_correlation


@MODELS.register_module()
class CorrelationHead(BaseModule):
    """Correlation head module.

    This module is proposed in
    "SiamRPN++: Evolution of Siamese Visual Tracking with Very Deep Networks.
    `SiamRPN++ <https://arxiv.org/abs/1812.11703>`_.

    Args:
        in_channels (int): Input channels.
        mid_channels (int): Middle channels.
        out_channels (int): Output channels.
        kernel_size (int): Kernel size of convs. Defaults to 3.
        norm_cfg (dict): Configuration of normlization method after each conv.
            Defaults to dict(type='BN').
        act_cfg (dict): Configuration of activation method after each conv.
            Defaults to dict(type='ReLU').
        init_cfg (Optional(dict)): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 mid_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 norm_cfg: dict = dict(type='BN'),
                 act_cfg: dict = dict(type='ReLU'),
                 init_cfg: Optional[dict] = None,
                 **kwargs):
        super(CorrelationHead, self).__init__(init_cfg)
        self.kernel_convs = ConvModule(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.search_convs = ConvModule(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.head_convs = nn.Sequential(
            ConvModule(
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_size=1,
                act_cfg=None))

    def forward(self, kernel: Tensor, search: Tensor) -> Tensor:
        """Forward function.

        Args:
            kernel (Tensor): The feature map of the template.
            search (Tensor): The feature map of search images.

        Returns:
            Tensor: The correlation results.
        """
        kernel = self.kernel_convs(kernel)
        search = self.search_convs(search)
        correlation_maps = depthwise_correlation(search, kernel)
        out = self.head_convs(correlation_maps)
        return out


@MODELS.register_module()
class SiameseRPNHead(BaseModule):
    """Siamese RPN head.

    This module is proposed in
    "SiamRPN++: Evolution of Siamese Visual Tracking with Very Deep Networks.
    `SiamRPN++ <https://arxiv.org/abs/1812.11703>`_.

    Args:
        anchor_generator (dict): Configuration to build anchor generator
            module.

        in_channels (int): Input channels.

        kernel_size (int): Kernel size of convs. Defaults to 3.

        norm_cfg (dict): Configuration of normlization method after each conv.
            Defaults to dict(type='BN').

        weighted_sum (bool): If True, use learnable weights to weightedly sum
            the output of multi heads in siamese rpn , otherwise, use
            averaging. Defaults to False.

        bbox_coder (dict): Configuration to build bbox coder. Defaults to
            dict(type='DeltaXYWHBBoxCoder', target_means=[0., 0., 0., 0.],
            target_stds=[1., 1., 1., 1.]).

        loss_cls (dict): Configuration to build classification loss. Defaults
            to dict( type='CrossEntropyLoss', reduction='sum', loss_weight=1.0)

        loss_bbox (dict): Configuration to build bbox regression loss. Defaults
            to dict( type='L1Loss', reduction='sum', loss_weight=1.2).

        train_cfg (Dict): Training setting. Defaults to None.

        test_cfg (Dict): Testing setting. Defaults to None.

        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 anchor_generator: dict,
                 in_channels: int,
                 kernel_size: int = 3,
                 norm_cfg: dict = dict(type='BN'),
                 weighted_sum: bool = False,
                 bbox_coder: dict = dict(
                     type='DeltaXYWHBBoxCoder',
                     target_means=[0., 0., 0., 0.],
                     target_stds=[1., 1., 1., 1.]),
                 loss_cls: dict = dict(
                     type='CrossEntropyLoss', reduction='sum',
                     loss_weight=1.0),
                 loss_bbox: dict = dict(
                     type='L1Loss', reduction='sum', loss_weight=1.2),
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None,
                 *args,
                 **kwargs):
        super(SiameseRPNHead, self).__init__(init_cfg)
        self.anchor_generator = TASK_UTILS.build(anchor_generator)
        self.bbox_coder = TASK_UTILS.build(bbox_coder)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.assigner = TASK_UTILS.build(self.train_cfg.assigner)
        self.sampler = TASK_UTILS.build(self.train_cfg.sampler)
        self.fp16_enabled = False

        self.cls_heads = nn.ModuleList()
        self.reg_heads = nn.ModuleList()
        for i in range(len(in_channels)):
            self.cls_heads.append(
                CorrelationHead(in_channels[i], in_channels[i],
                                2 * self.anchor_generator.num_base_anchors[0],
                                kernel_size, norm_cfg))
            self.reg_heads.append(
                CorrelationHead(in_channels[i], in_channels[i],
                                4 * self.anchor_generator.num_base_anchors[0],
                                kernel_size, norm_cfg))

        self.weighted_sum = weighted_sum
        if self.weighted_sum:
            self.cls_weight = nn.Parameter(torch.ones(len(in_channels)))
            self.reg_weight = nn.Parameter(torch.ones(len(in_channels)))

        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)

    def forward(self, template_feats: Tuple[Tensor, ...],
                search_feats: Tuple[Tensor, ...]) -> Tuple[Tensor, Tensor]:
        """Forward with features `template_feats` of template images and
        features `search_feats` of search images.

        Args:
            template_feats (tuple[Tensor, ...]): Tuple of Tensor with shape
                (N, C, H, W) denoting the multi level feature maps of template
                images. Typically H and W equal to 7.
            search_feats (tuple[Tensor, ...]): Tuple of Tensor with shape
                (N, C, H, W) denoting the multi level feature maps of search
                images. Typically H and W equal to 31.

        Returns:
            tuple(Tensor, Tensor): It contains
              - ``cls_score``: a Tensor with shape
                (N, 2 * num_base_anchors, H, W)
              - ``bbox_pred``: a Tensor with shape
                (N, 4 * num_base_anchors, H, W).
                Typically H and W equal to 25.
        """
        assert isinstance(template_feats, tuple) and isinstance(
            search_feats, tuple)
        assert len(template_feats) == len(search_feats) and len(
            template_feats) == len(self.cls_heads)

        if self.weighted_sum:
            cls_weight = nn.functional.softmax(self.cls_weight, dim=0)
            reg_weight = nn.functional.softmax(self.reg_weight, dim=0)
        else:
            reg_weight = cls_weight = [
                1.0 / len(template_feats) for i in range(len(template_feats))
            ]

        cls_score = 0
        bbox_pred = 0
        for i in range(len(template_feats)):
            cls_score_single = self.cls_heads[i](template_feats[i],
                                                 search_feats[i])
            bbox_pred_single = self.reg_heads[i](template_feats[i],
                                                 search_feats[i])
            cls_score += cls_weight[i] * cls_score_single
            bbox_pred += reg_weight[i] * bbox_pred_single

        return cls_score, bbox_pred

    def _get_init_targets(self, bboxes: Tensor,
                          score_maps_size: torch.Size) -> Tuple[Tensor, ...]:
        """Initialize the training targets based on flattened anchors of the
        last score map.

        Args:
            bboxes (Tensor): The generated anchors.
            score_maps_size (torch.Size): denoting the output size
                (height, width) of the network.

        Returns:
            tuple(Tensor, ...): It contains
              - ``labels``: in shape (N, H * W * num_base_anchors)
              - ``labels_weights``: in shape (N, H * W * num_base_anchors)
              - ``bbox_targets``: in shape (N, H * W * num_base_anchors, 4)
              - ``bbox_weights``: in shape (N, H * W * num_base_anchors, 4)
        """
        num_base_anchors = self.anchor_generator.num_base_anchors[0]
        H, W = score_maps_size
        num_anchors = H * W * num_base_anchors
        labels = bboxes.new_zeros((num_anchors, ), dtype=torch.long)
        labels_weights = bboxes.new_zeros((num_anchors, ))
        bbox_weights = bboxes.new_zeros((num_anchors, 4))
        bbox_targets = bboxes.new_zeros((num_anchors, 4))
        return labels, labels_weights, bbox_targets, bbox_weights

    def _get_positive_pair_targets(
            self, gt_instances: InstanceData,
            score_maps_size: torch.Size) -> Tuple[Tensor, ...]:
        """Generate the training targets for positive exemplar image and search
        image pair.

        Args:
            gt_instances (:obj:`InstanceData`): Groundtruth instances. It
                usually includes ``bboxes`` and ``labels`` attributes.
                ``bboxes`` of each search image is of
                shape (1, 4) in [tl_x, tl_y, br_x, br_y] format.
            score_maps_size (torch.Size): denoting the output size
                (height, width) of the network.

        Returns:
            tuple(Tensor, ...): It contains
              - ``labels``: in shape (N, H * W * num_base_anchors)
              - ``labels_weights``: in shape (N, H * W * num_base_anchors)
              - ``bbox_targets``: in shape (N, H * W * num_base_anchors, 4)
              - ``bbox_weights``: in shape (N, H * W * num_base_anchors, 4)
        """
        gt_bbox = gt_instances.bboxes
        (labels, labels_weights, bbox_targets,
         bbox_weights) = self._get_init_targets(gt_bbox, score_maps_size)

        if not hasattr(self, 'anchors'):
            self.anchors = self.anchor_generator.grid_priors(
                [score_maps_size], device=gt_bbox.device)[0]
            # Transform the coordinate origin from the top left corner to the
            # center in the scaled score map.
            feat_h, feat_w = score_maps_size
            stride_w, stride_h = self.anchor_generator.strides[0]
            self.anchors[:, 0:4:2] -= (feat_w // 2) * stride_w
            self.anchors[:, 1:4:2] -= (feat_h // 2) * stride_h

        anchors = self.anchors.clone()

        # The scaled feature map and the searched image have the same center.
        # Transform coordinate origin from the center to the top left corner in
        # the searched image.
        anchors += self.train_cfg.search_size // 2

        pred_instances = InstanceData(priors=anchors)
        assign_result = self.assigner.assign(pred_instances, gt_instances)
        sampling_result = self.sampler.sample(assign_result, pred_instances,
                                              gt_instances)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        neg_upper_bound = int(self.sampler.num *
                              (1 - self.sampler.pos_fraction))
        if len(neg_inds) > neg_upper_bound:
            neg_inds = neg_inds[:neg_upper_bound]

        if len(pos_inds) > 0:
            labels[pos_inds] = 1
            labels_weights[pos_inds] = 1.0 / len(pos_inds) / 2
            bbox_weights[pos_inds] = 1.0 / len(pos_inds)

        if len(neg_inds) > 0:
            labels[neg_inds] = 0
            labels_weights[neg_inds] = 1.0 / len(neg_inds) / 2

        bbox_targets[pos_inds, :] = self.bbox_coder.encode(
            sampling_result.pos_priors, sampling_result.pos_gt_bboxes)

        return labels, labels_weights, bbox_targets, bbox_weights

    def _get_negative_pair_targets(
            self, gt_instances: InstanceData,
            score_maps_size: torch.Size) -> Tuple[Tensor, ...]:
        """Generate the training targets for negative exemplar image and search
        image pair.

        Args:
            gt_instances (:obj:`InstanceData`): Groundtruth instances. It
                usually includes ``bboxes`` and ``labels`` attributes.
                ``bboxes`` of each search image is of
                shape (1, 4) in [tl_x, tl_y, br_x, br_y] format.
            score_maps_size (torch.Size): denoting the output size
                (height, width) of the network.

        Returns:
            tuple(Tensor, ...): It contains
              - ``labels``: in shape (N, H * W * num_base_anchors)
              - ``labels_weights``: in shape (N, H * W * num_base_anchors)
              - ``bbox_targets``: in shape (N, H * W * num_base_anchors, 4)
              - ``bbox_weights``: in shape (N, H * W * num_base_anchors, 4)
        """
        gt_bbox = gt_instances.bboxes
        (labels, labels_weights, bbox_targets,
         bbox_weights) = self._get_init_targets(gt_bbox, score_maps_size)
        H, W = score_maps_size
        target_cx, target_cy, _, _ = bbox_xyxy_to_cxcywh(gt_bbox)[0]
        anchor_stride = self.anchor_generator.strides[0]

        cx = W // 2
        cy = H // 2
        cx += int(
            torch.ceil((target_cx - self.train_cfg.search_size // 2) /
                       anchor_stride[0] + 0.5))
        cy += int(
            torch.ceil((target_cy - self.train_cfg.search_size // 2) /
                       anchor_stride[1] + 0.5))

        left = max(0, cx - 3)
        right = min(W, cx + 4)
        top = max(0, cy - 3)
        down = min(H, cy + 4)

        labels = labels.view(H, W, -1)
        labels[...] = -1
        labels[top:down, left:right, :] = 0

        labels = labels.view(-1)
        neg_inds = torch.nonzero(labels == 0, as_tuple=False)[:, 0]
        index = torch.randperm(
            neg_inds.numel(), device=neg_inds.device)[:self.train_cfg.num_neg]
        neg_inds = neg_inds[index]

        labels[...] = -1
        if len(neg_inds) > 0:
            labels[neg_inds] = 0
            labels_weights[neg_inds] = 1.0 / len(neg_inds) / 2

        # TODO: check it whether it's right.
        labels[...] = 0

        return labels, labels_weights, bbox_targets, bbox_weights

    def get_targets(self, batch_gt_instances: InstanceList,
                    score_maps_size: torch.Size) -> Tuple[Tensor, ...]:
        """Generate the training targets for exemplar image and search image
        pairs.

        Args:
            batch_gt_instances (list[InstanceData]): Batch of
                groundtruth instances. It usually includes ``bboxes`` and
                ``labels`` attributes. ``bboxes`` of each search image is of
                shape (1, 4) in [tl_x, tl_y, br_x, br_y] format.
            score_maps_size (torch.Size): denoting the output size
                (height, width) of the network.

        Returns:
            tuple(Tensor, ...): It contains
              - ``all_labels``: in shape (N, H * W * num_base_anchors)
              - ``all_labels_weights``: in shape (N, H * W * num_base_anchors)
              - ``all_bbox_targets``: in shape (N, H * W * num_base_anchors, 4)
              - ``all_bbox_weights``: in shape (N, H * W * num_base_anchors, 4)
        """
        (all_labels, all_labels_weights, all_bbox_targets,
         all_bbox_weights) = [], [], [], []

        for gt_instances in batch_gt_instances:
            is_positive_pair = gt_instances['labels'][0]
            if is_positive_pair:
                (labels, labels_weights, bbox_targets,
                 bbox_weights) = self._get_positive_pair_targets(
                     gt_instances, score_maps_size)
            else:
                (labels, labels_weights, bbox_targets,
                 bbox_weights) = self._get_negative_pair_targets(
                     gt_instances, score_maps_size)

            all_labels.append(labels)
            all_labels_weights.append(labels_weights)
            all_bbox_targets.append(bbox_targets)
            all_bbox_weights.append(bbox_weights)

        all_labels = torch.stack(all_labels)
        all_labels_weights = torch.stack(all_labels_weights) / len(
            all_labels_weights)
        all_bbox_targets = torch.stack(all_bbox_targets)
        all_bbox_weights = torch.stack(all_bbox_weights) / len(
            all_bbox_weights)

        return (all_labels, all_labels_weights, all_bbox_targets,
                all_bbox_weights)

    def loss(self, template_feats: Tuple[Tensor], search_feats: Tuple[Tensor],
             data_samples: SampleList, **kwargs) -> dict:
        """Perform forward propagation and loss calculation of the tracking
        head on the features of the upstream network.

        Args:
            template_feats (tuple[Tensor, ...]): Tuple of Tensor with
                shape (N, C, H, W) denoting the multi level feature maps of
                exemplar images. Typically H and W equal to 7.
            search_feats (tuple[Tensor, ...]): Tuple of Tensor with shape
                (N, C, H, W) denoting the multi level feature maps of
                search images. Typically H and W equal to 31.
            data_samples (List[:obj:`TrackDataSample`]): The Data
                Samples. It usually includes information such as `gt_instance`.

        Returns:
            dict: A dictionary of loss components.
        """
        outs = self(template_feats, search_feats)

        batch_gt_instances = []
        batch_img_metas = []
        batch_search_gt_instances = []
        for data_sample in data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)
            batch_search_gt_instances.append(data_sample.search_gt_instances)

        loss_inputs = outs + (batch_gt_instances, batch_search_gt_instances,
                              batch_img_metas)
        losses = self.loss_by_feat(*loss_inputs)
        return losses

    def loss_by_feat(self, cls_score: Tensor, bbox_pred: Tensor,
                     batch_gt_instances: InstanceList,
                     batch_search_gt_instances: InstanceList,
                     batch_img_metas: List[dict]) -> dict:
        """Compute loss.

        Args:
            cls_score (Tensor): of shape (N, 2 * num_base_anchors, H, W).
            bbox_pred (Tensor): of shape (N, 4 * num_base_anchors, H, W).
            batch_gt_instances (InstanceList): the instances in a batch.
            batch_search_gt_instances (InstanceList): the search instances in a
                batch.
            batch_img_metas (List[dict]): the meta information of all images in
                a batch.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        N, _, H, W = cls_score.shape
        (labels, labels_weights, bbox_targets,
         bbox_weights) = self.get_targets(batch_search_gt_instances, (H, W))

        losses = {}

        cls_score = cls_score.view(N, 2, -1, H, W)
        cls_score = cls_score.permute(0, 3, 4, 2, 1).contiguous().view(-1, 2)
        labels = labels.view(-1)
        labels_weights = labels_weights.view(-1)
        losses['loss_rpn_cls'] = self.loss_cls(
            cls_score, labels, weight=labels_weights)

        bbox_pred = bbox_pred.view(N, 4, -1, H, W)
        bbox_pred = bbox_pred.permute(0, 3, 4, 2, 1).contiguous().view(-1, 4)

        bbox_targets = bbox_targets.view(-1, 4)
        bbox_weights = bbox_weights.view(-1, 4)
        losses['loss_rpn_bbox'] = self.loss_bbox(
            bbox_pred, bbox_targets, weight=bbox_weights)

        return losses

    def predict(self, template_feats: Tuple[Tensor],
                search_feats: Tuple[Tensor], data_samples: SampleList,
                prev_bbox: Tensor, scale_factor: Tensor) -> InstanceList:
        """Perform forward propagation of the tracking head and predict
        tracking results on the features of the upstream network.

        Args:
            template_feats (tuple[Tensor, ...]): Tuple of Tensor with
                shape (N, C, H, W) denoting the multi level feature maps of
                exemplar images. Typically H and W equal to 7.
            search_feats (tuple[Tensor, ...]): Tuple of Tensor with shape
                (N, C, H, W) denoting the multi level feature maps of
                search images. Typically H and W equal to 31.

            data_samples (List[:obj:`TrackDataSample`]): The Data
                Samples. It usually includes information such as `gt_instance`.
            prev_bbox (Tensor): of shape (4, ) in [cx, cy, w, h] format.
            scale_factor (Tensor): scale factor.

        Returns:
            List[:obj:`InstanceData`]: Object tracking results of each image
            after the post process. Each item usually contains following keys.
                - scores (Tensor): Classification scores, has a shape (1, )
                - bboxes (Tensor): Has a shape (1, 4),
                  the last dimension 4 arrange as [x1, y1, x2, y2].
        """
        batch_img_metas = [
            data_samples.metainfo for data_samples in data_samples
        ]
        outs = self(template_feats, search_feats)
        predictions = self.predict_by_feat(
            *outs,
            prev_bbox=prev_bbox,
            scale_factor=scale_factor,
            batch_img_metas=batch_img_metas)
        return predictions

    def predict_by_feat(self, cls_score: Tensor, bbox_pred: Tensor,
                        prev_bbox: Tensor, scale_factor: Tensor,
                        batch_img_metas: List[dict]) -> InstanceList:
        """Track `prev_bbox` to current frame based on the output of network.

        Args:
            cls_score (Tensor): of shape (1, 2 * num_base_anchors, H, W).
            bbox_pred (Tensor): of shape (1, 4 * num_base_anchors, H, W).
            prev_bbox (Tensor): of shape (4, ) in [cx, cy, w, h] format.
            scale_factor (Tensor): scale factor.
            batch_img_metas (List[dict]): the meta information of all images in
                a batch.

        Returns:
            List[:obj:`InstanceData`]: Object tracking results of each image
            after the post process. Each item usually contains following keys.
                - scores (Tensor): Classification scores, has a shape (1, )
                - bboxes (Tensor): Has a shape (1, 4),
                  the last dimension 4 arrange as [x1, y1, x2, y2].
        """
        score_maps_size = [(cls_score.shape[2:])]
        if not hasattr(self, 'anchors'):
            self.anchors = self.anchor_generator.grid_priors(
                score_maps_size, device=cls_score.device)[0]
            # Transform the coordinate origin from the top left corner to the
            # center in the scaled feature map.
            feat_h, feat_w = score_maps_size[0]
            stride_w, stride_h = self.anchor_generator.strides[0]
            self.anchors[:, 0:4:2] -= (feat_w // 2) * stride_w
            self.anchors[:, 1:4:2] -= (feat_h // 2) * stride_h

        if not hasattr(self, 'windows'):
            self.windows = self.anchor_generator.gen_2d_hanning_windows(
                score_maps_size, cls_score.device)[0]

        H, W = score_maps_size[0]
        cls_score = cls_score.view(2, -1, H, W)
        cls_score = cls_score.permute(2, 3, 1, 0).contiguous().view(-1, 2)
        cls_score = cls_score.softmax(dim=1)[:, 1]

        bbox_pred = bbox_pred.view(4, -1, H, W)
        bbox_pred = bbox_pred.permute(2, 3, 1, 0).contiguous().view(-1, 4)
        bbox_pred = self.bbox_coder.decode(self.anchors, bbox_pred)

        result = InstanceData()
        result.scores = cls_score
        result.bboxes = bbox_pred

        return self._bbox_post_process([result],
                                       prev_bbox,
                                       scale_factor,
                                       batch_img_metas=batch_img_metas)

    def _bbox_post_process(self, results: InstanceList, prev_bbox: Tensor,
                           scale_factor: Tensor, batch_img_metas: List[dict],
                           **kwargs) -> InstanceList:
        """The postprocess of tracking bboxes.

        Args:
            results (InstanceList): tracking results.
            prev_bbox (Tensor): of shape (4, ) in [cx, cy, w, h] format.
            scale_factor (Tensor): scale factor.
            batch_img_metas (List[dict]): the meta information of all images in
                a batch.

        Returns:
            List[:obj:`InstanceData`]: Object tracking results of each image
            after the post process. Each item usually contains following keys.
                - scores (Tensor): Classification scores, has a shape (1, )
                - bboxes (Tensor): Has a shape (1, 4),
                  the last dimension 4 arrange as [x1, y1, x2, y2].
        """
        result = results[0]
        bbox_pred = bbox_xyxy_to_cxcywh(result.bboxes)

        def change_ratio(ratio):
            return torch.max(ratio, 1. / ratio)

        def enlarge_size(w, h):
            pad = (w + h) * 0.5
            return torch.sqrt((w + pad) * (h + pad))

        # scale penalty
        scale_penalty = change_ratio(
            enlarge_size(bbox_pred[:, 2], bbox_pred[:, 3]) / enlarge_size(
                prev_bbox[2] * scale_factor, prev_bbox[3] * scale_factor))

        # aspect ratio penalty
        aspect_ratio_penalty = change_ratio(
            (prev_bbox[2] / prev_bbox[3]) /
            (bbox_pred[:, 2] / bbox_pred[:, 3]))

        # penalize cls_score
        penalty = torch.exp(-(aspect_ratio_penalty * scale_penalty - 1) *
                            self.test_cfg.penalty_k)
        penalty_score = penalty * result.scores

        # window penalty
        penalty_score = penalty_score * (1 - self.test_cfg.window_influence) \
            + self.windows * self.test_cfg.window_influence

        best_idx = torch.argmax(penalty_score)
        result = result[best_idx.item()]

        best_bbox = bbox_pred[best_idx, :] / scale_factor
        final_bbox = torch.zeros_like(best_bbox)

        # map the bbox center from the searched image to the original image.
        final_bbox[0] = best_bbox[0] + prev_bbox[0]
        final_bbox[1] = best_bbox[1] + prev_bbox[1]

        # smooth bbox
        lr = penalty[best_idx] * result.scores * self.test_cfg.lr
        final_bbox[2] = prev_bbox[2] * (1 - lr) + best_bbox[2] * lr
        final_bbox[3] = prev_bbox[3] * (1 - lr) + best_bbox[3] * lr

        # clip boundary
        img_shape = batch_img_metas[0]['ori_shape']
        # rather than [x1, x2, y1, y2] format.
        final_bbox = self._bbox_clip(final_bbox, img_shape[0], img_shape[1])

        result.bboxes = bbox_cxcywh_to_xyxy(final_bbox)[None]

        return [result]

    def _bbox_clip(self, bbox: Tensor, img_h: int, img_w: int) -> Tensor:
        """Clip the bbox with [cx, cy, w, h] format.

        Args:
            img (Tensor): of shape (1, C, H, W) encoding original input
                image.
            bbox (Tensor): The given instance bbox of first frame that
                need be tracked in the following frames. The shape of the box
                is of (4, ) shape in [cx, cy, w, h] format.

        Returns:
            Tensor: The clipped boxes.
        """
        bbox[0] = bbox[0].clamp(0., img_w)
        bbox[1] = bbox[1].clamp(0., img_h)
        bbox[2] = bbox[2].clamp(10., img_w)
        bbox[3] = bbox[3].clamp(10., img_h)
        return bbox
