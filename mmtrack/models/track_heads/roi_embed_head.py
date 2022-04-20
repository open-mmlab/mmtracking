# Copyright (c) OpenMMLab. All rights reserved.
from collections import defaultdict

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from mmdet.models import HEADS, build_loss
from mmdet.models.losses import accuracy
from torch.nn.modules.utils import _pair

from mmtrack.core import embed_similarity


@HEADS.register_module()
class RoIEmbedHead(BaseModule):
    """The roi embed head.

    This module is used in multi-object tracking methods, such as MaskTrack
    R-CNN.

    Args:
        num_convs (int): The number of convoluational layers to embed roi
            features. Defaults to 0.
        num_fcs (int): The number of fully connection layers to embed roi
            features. Defaults to 0.
        roi_feat_size (int|tuple(int)): The spatial size of roi features.
            Defaults to 7.
        in_channels (int): The input channel of roi features. Defaults to 256.
        conv_out_channels (int): The output channel of roi features after
            forwarding convoluational layers. Defaults to 256.
        with_avg_pool (bool): Whether use average pooling before passing roi
            features into fully connection layers. Defaults to False.
        fc_out_channels (int): The output channel of roi features after
            forwarding fully connection layers. Defaults to 1024.
        conv_cfg (dict): Config dict for convolution layer. Defaults to None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer. Defaults to None.
        loss_match (dict): The loss function. Defaults to
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
        init_cfg (dict): Configuration of initialization. Defaults to None.
    """

    def __init__(self,
                 num_convs=0,
                 num_fcs=0,
                 roi_feat_size=7,
                 in_channels=256,
                 conv_out_channels=256,
                 with_avg_pool=False,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_match=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 init_cfg=None,
                 **kwargs):
        super(RoIEmbedHead, self).__init__(init_cfg=init_cfg)
        self.num_convs = num_convs
        self.num_fcs = num_fcs
        self.roi_feat_size = _pair(roi_feat_size)
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.with_avg_pool = with_avg_pool
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.loss_match = build_loss(loss_match)
        self.fp16_enabled = False

        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(self.roi_feat_size)
        # add convs and fcs
        self.convs, self.fcs, self.last_layer_dim = self._add_conv_fc_branch(
            self.num_convs, self.num_fcs, self.in_channels)
        self.relu = nn.ReLU(inplace=True)

    def _add_conv_fc_branch(self, num_branch_convs, num_branch_fcs,
                            in_channels):
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels

        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            if not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels

        return branch_convs, branch_fcs, last_layer_dim

    @property
    def custom_activation(self):
        return getattr(self.loss_match, 'custom_activation', False)

    def _forward(self, x, num_x_per_img):
        """Forward the input `x`, and split the output to a list.

        Args:
            x (Tensor): of shape [N, C, H, W]. N is the number of proposals.
            num_x_per_img (list[int]): The `x` contains proposals of
                multi-images. `num_x_per_img` denotes the number of proposals
                for each image.

        Returns:
            list[Tensor]: Each Tensor denotes the embed features belonging to
            an image in a batch.
        """
        if self.num_convs > 0:
            for conv in self.convs:
                x = conv(x)

        if self.num_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            x = x.flatten(1)
            for fc in self.fcs:
                x = self.relu(fc(x))
        else:
            x = x.flatten(1)

        x_split = torch.split(x, num_x_per_img, dim=0)
        return x_split

    @auto_fp16(apply_to=('x', 'ref_x'))
    def forward(self, x, ref_x, num_x_per_img, num_x_per_ref_img):
        """Computing the similarity scores between `x` and `ref_x`.

        Args:
            x (Tensor): of shape [N, C, H, W]. N is the number of key frame
                proposals.
            ref_x (Tensor): of shape [M, C, H, W]. M is the number of reference
                frame proposals.
            num_x_per_img (list[int]): The `x` contains proposals of
                multi-images. `num_x_per_img` denotes the number of proposals
                for each key image.
            num_x_per_ref_img (list[int]): The `ref_x` contains proposals of
                multi-images. `num_x_per_ref_img` denotes the number of
                proposals for each reference image.

        Returns:
            list[Tensor]: The predicted similarity_logits of each pair of key
            image and reference image.
        """
        x_split = self._forward(x, num_x_per_img)
        ref_x_split = self._forward(ref_x, num_x_per_ref_img)

        similarity_logits = []
        for one_x, one_ref_x in zip(x_split, ref_x_split):
            similarity_logit = embed_similarity(
                one_x, one_ref_x, method='dot_product')
            dummy = similarity_logit.new_zeros(one_x.shape[0], 1)
            similarity_logit = torch.cat((dummy, similarity_logit), dim=1)
            similarity_logits.append(similarity_logit)
        return similarity_logits

    def get_targets(self, sampling_results, gt_instance_ids,
                    ref_gt_instance_ids):
        """Calculate the ground truth for all samples in a batch according to
        the sampling_results.

        Args:
            sampling_results (List[obj:SamplingResults]): Assign results of
                all images in a batch after sampling.
            gt_instance_ids (list[Tensor]): The instance ids of gt_bboxes of
                all images in a batch, each tensor has shape (num_gt, ).
            ref_gt_instance_ids (list[Tensor]): The instance ids of gt_bboxes
                of all reference images in a batch, each tensor has shape
                (num_gt, ).

        Returns:
            Tuple[list[Tensor]]: Ground truth for proposals in a batch.
            Containing the following list of Tensors:

                - track_id_targets (list[Tensor]): The instance ids of
                  Gt_labels for all proposals in a batch, each tensor in list
                  has shape (num_proposals,).
                - track_id_weights (list[Tensor]): Labels_weights for
                  all proposals in a batch, each tensor in list has
                  shape (num_proposals,).
        """
        track_id_targets = []
        track_id_weights = []

        for res, gt_instance_id, ref_gt_instance_id in zip(
                sampling_results, gt_instance_ids, ref_gt_instance_ids):
            pos_instance_ids = gt_instance_id[res.pos_assigned_gt_inds]
            pos_match_id = gt_instance_id.new_zeros(len(pos_instance_ids))
            for i, id in enumerate(pos_instance_ids):
                if id in ref_gt_instance_id:
                    pos_match_id[i] = ref_gt_instance_id.tolist().index(id) + 1

            track_id_target = gt_instance_id.new_zeros(
                len(res.bboxes), dtype=torch.int64)
            track_id_target[:len(res.pos_bboxes)] = pos_match_id
            track_id_weight = res.bboxes.new_zeros(len(res.bboxes))
            track_id_weight[:len(res.pos_bboxes)] = 1.0

            track_id_targets.append(track_id_target)
            track_id_weights.append(track_id_weight)

        return track_id_targets, track_id_weights

    @force_fp32(apply_to=('similarity_logits', ))
    def loss(self,
             similarity_logits,
             track_id_targets,
             track_id_weights,
             reduction_override=None):
        """Calculate the loss in a batch.

        Args:
            similarity_logits (list[Tensor]): The predicted similarity_logits
                of each pair of key image and reference image.
            track_id_targets (list[Tensor]): The instance ids of Gt_labels for
                all proposals in a batch, each tensor in list has shape
                (num_proposals,).
            track_id_weights (list[Tensor]): Labels_weights for
                all proposals in a batch, each tensor in list has shape
                (num_proposals,).
            reduction_override (str, optional): The method used to reduce the
                loss. Options are "none", "mean" and "sum".

        Returns:
            dict[str, Tensor]: a dictionary of loss components.
        """
        assert isinstance(similarity_logits, list)
        assert isinstance(track_id_targets, list)
        assert isinstance(track_id_weights, list)
        assert len(similarity_logits) == len(track_id_targets)
        assert len(track_id_weights) == len(track_id_targets)
        losses = defaultdict(list)

        for similarity_logit, track_id_target, track_id_weight in zip(
                similarity_logits, track_id_targets, track_id_weights):
            avg_factor = max(torch.sum(track_id_target > 0).float().item(), 1.)
            if similarity_logit.numel() > 0:
                loss_match = self.loss_match(
                    similarity_logit,
                    track_id_target,
                    track_id_weight,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                if isinstance(loss_match, dict):
                    for key, value in loss_match.items():
                        losses[key].append(value)
                else:
                    losses['loss_match'].append(loss_match)

                valid_index = track_id_weight > 0
                valid_similarity_logit = similarity_logit[valid_index]
                valid_track_id_target = track_id_target[valid_index]
                if self.custom_activation:
                    match_accuracy = self.loss_match.get_accuracy(
                        valid_similarity_logit, valid_track_id_target)
                    for key, value in match_accuracy.items():
                        losses[key].append(value)
                else:
                    losses['match_accuracy'].append(
                        accuracy(valid_similarity_logit,
                                 valid_track_id_target))

        for key, value in losses.items():
            losses[key] = sum(losses[key]) / len(similarity_logits)
        return losses
