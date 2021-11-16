# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn.bricks import ConvModule
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from mmdet.core import build_assigner, build_bbox_coder, build_sampler
from mmdet.core.anchor import build_prior_generator
from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh
from mmdet.models import HEADS, build_loss

from mmtrack.core.track import depthwise_correlation


@HEADS.register_module()
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
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 kernel_size=3,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None,
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

    def forward(self, kernel, search):
        kernel = self.kernel_convs(kernel)
        search = self.search_convs(search)
        correlation_maps = depthwise_correlation(search, kernel)
        out = self.head_convs(correlation_maps)
        return out


@HEADS.register_module()
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
                 anchor_generator,
                 in_channels,
                 kernel_size=3,
                 norm_cfg=dict(type='BN'),
                 weighted_sum=False,
                 bbox_coder=dict(
                     type='DeltaXYWHBBoxCoder',
                     target_means=[0., 0., 0., 0.],
                     target_stds=[1., 1., 1., 1.]),
                 loss_cls=dict(
                     type='CrossEntropyLoss', reduction='sum',
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='L1Loss', reduction='sum', loss_weight=1.2),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 *args,
                 **kwargs):
        super(SiameseRPNHead, self).__init__(init_cfg)
        self.anchor_generator = build_prior_generator(anchor_generator)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.assigner = build_assigner(self.train_cfg.assigner)
        self.sampler = build_sampler(self.train_cfg.sampler)
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

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)

    @auto_fp16()
    def forward(self, z_feats, x_feats):
        """Forward with features `z_feats` of exemplar images and features
        `x_feats` of search images.

        Args:
            z_feats (tuple[Tensor]): Tuple of Tensor with shape (N, C, H, W)
                denoting the multi level feature maps of exemplar images.
                Typically H and W equal to 7.
            x_feats (tuple[Tensor]): Tuple of Tensor with shape (N, C, H, W)
                denoting the multi level feature maps of search images.
                Typically H and W equal to 31.

        Returns:
            tuple(cls_score, bbox_pred): cls_score is a Tensor with shape
            (N, 2 * num_base_anchors, H, W), bbox_pred is a Tensor with shape
            (N, 4 * num_base_anchors, H, W), Typically H and W equal to 25.
        """
        assert isinstance(z_feats, tuple) and isinstance(x_feats, tuple)
        assert len(z_feats) == len(x_feats) and len(z_feats) == len(
            self.cls_heads)

        if self.weighted_sum:
            cls_weight = nn.functional.softmax(self.cls_weight, dim=0)
            reg_weight = nn.functional.softmax(self.reg_weight, dim=0)
        else:
            reg_weight = cls_weight = [
                1.0 / len(z_feats) for i in range(len(z_feats))
            ]

        cls_score = 0
        bbox_pred = 0
        for i in range(len(z_feats)):
            cls_score_single = self.cls_heads[i](z_feats[i], x_feats[i])
            bbox_pred_single = self.reg_heads[i](z_feats[i], x_feats[i])
            cls_score += cls_weight[i] * cls_score_single
            bbox_pred += reg_weight[i] * bbox_pred_single

        return cls_score, bbox_pred

    def _get_init_targets(self, gt_bbox, score_maps_size):
        """Initialize the training targets based on flattened anchors of the
        last score map."""
        num_base_anchors = self.anchor_generator.num_base_anchors[0]
        H, W = score_maps_size
        num_anchors = H * W * num_base_anchors
        labels = gt_bbox.new_zeros((num_anchors, ), dtype=torch.long)
        labels_weights = gt_bbox.new_zeros((num_anchors, ))
        bbox_weights = gt_bbox.new_zeros((num_anchors, 4))
        bbox_targets = gt_bbox.new_zeros((num_anchors, 4))
        return labels, labels_weights, bbox_targets, bbox_weights

    def _get_positive_pair_targets(self, gt_bbox, score_maps_size):
        """Generate the training targets for positive exemplar image and search
        image pair.

        Args:
            gt_bbox (Tensor): Ground truth bboxes of an search image with
                shape (1, 5) in [0.0, tl_x, tl_y, br_x, br_y] format.
            score_maps_size (torch.size): denoting the output size
                (height, width) of the network.

        Returns:
            tuple(labels, labels_weights, bbox_targets, bbox_weights): the
            shape is (H * W * num_base_anchors,), (H * W * num_base_anchors,),
            (H * W * num_base_anchors, 4), (H * W * num_base_anchors, 4)
            respectively. All of them are Tensor.
        """
        (labels, labels_weights, _,
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

        assign_result = self.assigner.assign(anchors, gt_bbox[:, 1:])
        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bbox[:, 1:])
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

        bbox_targets = self.bbox_coder.encode(
            anchors, gt_bbox[:, 1:].repeat(anchors.shape[0], 1))
        return labels, labels_weights, bbox_targets, bbox_weights

    def _get_negative_pair_targets(self, gt_bbox, score_maps_size):
        """Generate the training targets for negative exemplar image and search
        image pair.

        Args:
            gt_bbox (Tensor): Ground truth bboxes of an search image with
                shape (1, 5) in [0.0, tl_x, tl_y, br_x, br_y] format.
            score_maps_size (torch.size): denoting the output size
                (height, width) of the network.

        Returns:
            tuple(labels, labels_weights, bbox_targets, bbox_weights): the
            shape is (H * W * num_base_anchors,), (H * W * num_base_anchors,),
            (H * W * num_base_anchors, 4), (H * W * num_base_anchors, 4)
            respectively. All of them are Tensor.
        """
        (labels, labels_weights, bbox_targets,
         bbox_weights) = self._get_init_targets(gt_bbox, score_maps_size)
        H, W = score_maps_size
        target_cx, target_cy, _, _ = bbox_xyxy_to_cxcywh(gt_bbox[:, 1:])[0]
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
        labels[...] = 0

        return labels, labels_weights, bbox_targets, bbox_weights

    def get_targets(self, gt_bboxes, score_maps_size, is_positive_pairs):
        """Generate the training targets for exemplar image and search image
        pairs.

        Args:
            gt_bboxes (list[Tensor]): Ground truth bboxes of each
                search image with shape (1, 5) in [0.0, tl_x, tl_y, br_x, br_y]
                format.
            score_maps_size (torch.size): denoting the output size
                (height, width) of the network.
            is_positive_pairs (bool): list of bool denoting whether each ground
                truth bbox in `gt_bboxes` is positive.

        Returns:
            tuple(all_labels, all_labels_weights, all_bbox_targets,
            all_bbox_weights): the shape is (N, H * W * num_base_anchors),
            (N, H * W * num_base_anchors), (N, H * W * num_base_anchors, 4),
            (N, H * W * num_base_anchors, 4), respectively. All of them are
            Tensor.
        """
        (all_labels, all_labels_weights, all_bbox_targets,
         all_bbox_weights) = [], [], [], []

        for gt_bbox, is_positive_pair in zip(gt_bboxes, is_positive_pairs):
            if is_positive_pair:
                (labels, labels_weights, bbox_targets,
                 bbox_weights) = self._get_positive_pair_targets(
                     gt_bbox, score_maps_size)
            else:
                (labels, labels_weights, bbox_targets,
                 bbox_weights) = self._get_negative_pair_targets(
                     gt_bbox, score_maps_size)

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

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self, cls_score, bbox_pred, labels, labels_weights, bbox_targets,
             bbox_weights):
        """Compute loss.

        Args:
            cls_score (Tensor): of shape (N, 2 * num_base_anchors, H, W).
            bbox_pred (Tensor): of shape (N, 4 * num_base_anchors, H, W).
            labels (Tensor): of shape (N, H * W * num_base_anchors).
            labels_weights (Tensor): of shape (N, H * W * num_base_anchors).
            bbox_targets (Tensor): of shape (N, H * W * num_base_anchors, 4).
            bbox_weights (Tensor): of shape (N, H * W * num_base_anchors, 4).

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        losses = {}
        N, _, H, W = cls_score.shape

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

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_bbox(self, cls_score, bbox_pred, prev_bbox, scale_factor):
        """Track `prev_bbox` to current frame based on the output of network.

        Args:
            cls_score (Tensor): of shape (1, 2 * num_base_anchors, H, W).
            bbox_pred (Tensor): of shape (1, 4 * num_base_anchors, H, W).
            prev_bbox (Tensor): of shape (4, ) in [cx, cy, w, h] format.
            scale_factor (Tensr): scale factor.

        Returns:
            tuple(best_score, best_bbox): best_score is a Tensor denoting the
            score of `best_bbox`, best_bbox is a Tensor of shape (4, )
            with [cx, cy, w, h] format, which denotes the best tracked
            bbox in current frame.
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
        bbox_pred = bbox_xyxy_to_cxcywh(bbox_pred)

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
        penalty_score = penalty * cls_score

        # window penalty
        penalty_score = penalty_score * (1 - self.test_cfg.window_influence) \
            + self.windows * self.test_cfg.window_influence

        best_idx = torch.argmax(penalty_score)
        best_score = cls_score[best_idx]
        best_bbox = bbox_pred[best_idx, :] / scale_factor

        final_bbox = torch.zeros_like(best_bbox)

        # map the bbox center from the searched image to the original image.
        final_bbox[0] = best_bbox[0] + prev_bbox[0]
        final_bbox[1] = best_bbox[1] + prev_bbox[1]

        # smooth bbox
        lr = penalty[best_idx] * cls_score[best_idx] * self.test_cfg.lr
        final_bbox[2] = prev_bbox[2] * (1 - lr) + best_bbox[2] * lr
        final_bbox[3] = prev_bbox[3] * (1 - lr) + best_bbox[3] * lr

        return best_score, final_bbox
