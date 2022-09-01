# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Tuple

import torch
import torch.nn as nn
from mmcv.cnn.bricks import ConvModule
from mmcv.ops import PrRoIPool
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy
from mmengine.model import BaseModule
from torch import Tensor

from mmtrack.registry import MODELS
from mmtrack.structures.bbox import (bbox_cxcywh_to_x1y1wh,
                                     bbox_rel_cxcywh_to_xywh,
                                     bbox_xywh_to_rel_cxcywh,
                                     bbox_xyxy_to_x1y1wh)
from mmtrack.utils import OptConfigType, SampleList


class LinearBlock(nn.Module):
    """The linear block. The full pipeline: FC > BN > ReLU.

    Args:
        in_planes (int): The dim of input.
        out_planes (int): The dim of output.
        input_size (int): The size of input.
        bias (bool, optional): Whether to have bias in linear layer. Defaults
            to True.
        batch_norm (bool, optional): Whether to have BN after linear layer.
            Defaults to True.
        relu (bool, optional):  Whether to have ReLU at last. Defaults to True.
    """

    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 input_size: int,
                 bias: bool = True,
                 batch_norm: bool = True,
                 relu: bool = True):
        super().__init__()
        self.linear = nn.Linear(
            in_planes * input_size * input_size, out_planes, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes) if batch_norm else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.linear(x.reshape(x.shape[0], -1))
        if self.bn is not None:
            x = self.bn(x.reshape(x.shape[0], x.shape[1], 1, 1))
        if self.relu is not None:
            x = self.relu(x)
        return x.reshape(x.shape[0], -1)


@MODELS.register_module()
class IouNetHead(BaseModule):
    """Module for IoU prediction.

    Refer to the ATOM paper for a detailed illustration of the architecture.
    `ATOM: <https://arxiv.org/abs/1811.07628>`_.

    Args:
        in_dim (tuple(int), optional): Feature dimensionality from the two
            input backbone layers. Defaults to (128, 256).
        pred_in_dim (tuple(int), optional): Input dimensionality of the
            prediction network. Defaults to (256, 256).
        pred_inter_dim (tuple(int), optional): Intermediate dimensionality in
            the prediction network. Defaults to (256, 256).
        bbox_cfg (dict, optional): The configuration of bbox refinement.
            Defaults to None. Defaults to None.
        train_cfg (dict, optional): The configuration of training.
            Defaults to None.
        loss_bbox (dict, optional): The configuration of loss.
            Defaults to None.
    """

    def __init__(self,
                 in_dim: Tuple = (128, 256),
                 pred_in_dim: Tuple = (256, 256),
                 pred_inter_dim: Tuple = (256, 256),
                 bbox_cfg: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 loss_bbox: OptConfigType = None,
                 **kwargs):
        super().__init__()
        self.bbox_cfg = bbox_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg

        self.loss_bbox = MODELS.build(loss_bbox)

        def conv_module(in_planes, out_planes, kernel_size=3, padding=1):
            # The module's pipeline: Conv -> BN -> ReLU.
            return ConvModule(
                in_channels=in_planes,
                out_channels=out_planes,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
                norm_cfg=dict(type='BN', requires_grad=True),
                act_cfg=dict(type='ReLU'),
                inplace=True)

        # `*_temp` denotes template branch, `*_search` denotes search branch
        # The `number` in the names of variables are block indexes in the
        # backbone or indexes of head layer.
        self.conv3_temp = conv_module(in_dim[0], 128)
        self.roi3_temp = PrRoIPool(3, 1 / 8)
        self.fc3_temp = conv_module(128, 256, padding=0)
        self.fc34_3_temp = conv_module(
            256 + 256, pred_in_dim[0], kernel_size=1, padding=0)

        self.conv4_temp = conv_module(in_dim[1], 256)
        self.roi4_temp = PrRoIPool(1, 1 / 16)
        self.fc34_4_temp = conv_module(
            256 + 256, pred_in_dim[1], kernel_size=1, padding=0)

        self.conv3_search = nn.Sequential(
            conv_module(in_dim[0], 256), conv_module(256, pred_in_dim[0]))
        self.roi3_search = PrRoIPool(5, 1 / 8)
        self.fc3_search = LinearBlock(pred_in_dim[0], pred_inter_dim[0], 5)

        self.conv4_search = nn.Sequential(
            conv_module(in_dim[1], 256), conv_module(256, pred_in_dim[1]))
        self.roi4_search = PrRoIPool(3, 1 / 16)
        self.fc4_search = LinearBlock(pred_in_dim[1], pred_inter_dim[1], 3)

        self.iou_predictor = nn.Linear(
            pred_inter_dim[0] + pred_inter_dim[1], 1, bias=True)

    def init_weights(self):
        """Initialize the parameters of this module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(
                    m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.uniform_()
                m.bias.data.zero_()

    def predict_iou(self, modulations: Tuple[Tensor], feats: Tensor,
                    proposals: Tensor) -> Tensor:
        """Predicts IoU for the give proposals.

        Args:
            modulations (Tuple(Tensor)): contains the features from two layers.
                The inner tensors are of shape (bs, c, 1, 1)
            feats (Tuple(Tensor)):  IoU features for test images.
                The inner tensors are of shape (bs, c, h, w).
            proposals (Tuple[Tensor]):  Proposal boxes for which the IoU will
                be predicted (bs, num_proposals, 4).

        Returns:
            Tensor: IoU between the proposals with the groundtruth boxes. It's
                of shape (bs, num_proposals).
        """
        # `*_temp` denotes template branch, `*_search` denotes search branch
        # The `number` in the names of variables are block indexes in the
        # backbone or indexes of head layer.
        fc34_3_temp, fc34_4_temp = modulations
        conv3_search, conv4_search = feats
        batch_size = conv3_search.shape[0]

        # Modulation
        conv3_search_att = conv3_search * fc34_3_temp
        conv4_search_att = conv4_search * fc34_4_temp

        # Push the different rois for the same image along the batch dimension
        num_proposals_per_batch = proposals.shape[1]

        # input proposals2 is in format xywh, convert it to x0y0x1y1 format
        proposals_xyxy = torch.cat(
            (proposals[..., 0:2], proposals[..., 0:2] + proposals[..., 2:4]),
            dim=2)

        # Add batch_index to rois
        batch_index = torch.arange(
            batch_size, dtype=torch.float32).reshape(-1, 1,
                                                     1).to(proposals.device)
        roi = torch.cat((batch_index.expand(-1, num_proposals_per_batch,
                                            -1), proposals_xyxy),
                        dim=2)
        roi = roi.reshape(-1, 5)

        roi3_search = self.roi3_search(conv3_search_att, roi)
        roi4_search = self.roi4_search(conv4_search_att, roi)

        fc3_search = self.fc3_search(roi3_search)
        fc4_search = self.fc4_search(roi4_search)

        fc34_search_cat = torch.cat((fc3_search, fc4_search), dim=1)

        iou_pred = self.iou_predictor(fc34_search_cat).reshape(
            batch_size, num_proposals_per_batch)

        return iou_pred

    def get_modulation(self, feats: Tuple[Tensor],
                       bboxes: Tensor) -> Tuple[Tensor, Tensor]:
        """Get modulation vectors for the targets in the search branch.

        Args:
            feats (tuple(Tensor)): Backbone features from template branch.
                It's of shape (bs, c, h, w).
            bboxes (Tensor): Target boxes (x1, y1, x2, y2) in image coords in
                the template branch. It's of shape (bs, 4).

        Returns:
            fc34_3_temp (Tensor): of shape (bs, c, 1, 1).
            fc34_4_temp (Tensor): of shape (bs, c, 1, 1).
        """

        # Add batch_index to rois
        batch_size = bboxes.shape[0]
        batch_index = torch.arange(
            batch_size, dtype=torch.float32).reshape(-1, 1).to(bboxes.device)
        roi = torch.cat((batch_index, bboxes), dim=1)

        # Perform conv and prpool on the feature maps from the backbone
        # `*_temp` denotes template branch, `*_search` denotes search branch
        # The `number` in the names of variables are block indexes in the
        # backbone or indexes of head layer.
        feat3_temp, feat4_temp = feats
        conv3_temp = self.conv3_temp(feat3_temp)
        roi3_temp = self.roi3_temp(conv3_temp, roi)
        fc3_temp = self.fc3_temp(roi3_temp)

        c4_temp = self.conv4_temp(feat4_temp)
        roi4_temp = self.roi4_temp(c4_temp, roi)

        # Concatenate from block 3 and 4
        fc34_temp = torch.cat((fc3_temp, roi4_temp), dim=1)

        fc34_3_temp = self.fc34_3_temp(fc34_temp)
        fc34_4_temp = self.fc34_4_temp(fc34_temp)

        return fc34_3_temp, fc34_4_temp

    def forward(self, feats: Tuple[Tensor]) -> Tuple[Tensor, Tensor]:
        """Get IoU prediction features from a 4 or 5 dimensional backbone
        input.

        Args:
            feats (tuple(Tensor)): Containing the features from backbone with
                shape (bs, c, h, w)

        Returns:
            conv3_search (Tensor): Features from the `conv3_search` branch.
            conv4_search (Tensor): Features from the `conv4_search` branch.
        """
        # `*_temp` denotes template branch, `*_search` denotes search branch
        # The `number` in the names of variables are block indexes in the
        # backbone or indexes of head layer.
        feats = [
            f.reshape(-1, *f.shape[-3:]) if f.dim() == 5 else f for f in feats
        ]
        feat3_search, feat4_search = feats
        conv3_search = self.conv3_search(feat3_search)
        conv4_search = self.conv4_search(feat4_search)

        return conv3_search, conv4_search

    def init_iou_net(self, iou_backbone_feats: Tensor, bboxes: Tensor):
        """Initialize the IoUNet with feature are from the 'layer2' and
        'layer3' of backbone.

        Args:
            iou_backbone_feats (tuple(Tensor)): The features from the backbone.
            bboxes (Tensor): of shape (4, ) or (1, 4) in [cx, cy, w, h] format.
        """
        bboxes = bbox_cxcywh_to_xyxy(bboxes.view(-1, 4))
        # Get modulation vector
        self.iou_modulation = self.get_modulation(iou_backbone_feats, bboxes)

    def optimize_bboxes(self, iou_features: Tuple[Tensor],
                        init_bboxes: Tensor) -> Tuple[Tensor, Tensor]:
        """Optimize the bboxes.

        Args:
            iou_features (tuple(Tensor)): The features used to predict IoU.
            init_bboxes (Tensor): The initialized bboxes with shape (N,4) in
                [cx, cy, w, h] format.

        Returns:
            Tensor: The optimized bboxes with shape (N,4)  in [x, y, w, h]
                format.
            Tensor: The predict IoU of the optimized bboxes with shape (N, ).
        """
        step_length = self.bbox_cfg['box_refine_step_length']
        if isinstance(step_length, (tuple, list)):
            step_length = torch.Tensor([
                step_length[0], step_length[0], step_length[1], step_length[1]
            ]).to(iou_features[0].device).view(1, 1, 4)

        # TODO: simplify this series of transform
        output_bboxes = bbox_cxcywh_to_x1y1wh(init_bboxes)
        output_bboxes = output_bboxes.view(1, -1, 4)
        bboxes_sz_norm = output_bboxes[:, :1, 2:].clone()
        output_bboxes_rel = bbox_xywh_to_rel_cxcywh(output_bboxes,
                                                    bboxes_sz_norm)

        with torch.set_grad_enabled(True):
            for _ in range(self.bbox_cfg['box_refine_iter']):
                # Forward
                bboxes_init_rel = output_bboxes_rel.clone().detach()
                bboxes_init_rel.requires_grad = True

                bboxes_init = bbox_rel_cxcywh_to_xywh(bboxes_init_rel,
                                                      bboxes_sz_norm)
                iou_outputs = self.predict_iou(self.iou_modulation,
                                               iou_features, bboxes_init)
                # Backward
                iou_outputs.backward(gradient=torch.ones_like(iou_outputs))

                # Update bboxes
                output_bboxes_rel = bboxes_init_rel + (
                    step_length * bboxes_init_rel.grad)
                output_bboxes_rel.detach_()

                step_length *= self.bbox_cfg['box_refine_step_decay']

        output_bboxes = bbox_rel_cxcywh_to_xywh(output_bboxes_rel,
                                                bboxes_sz_norm)

        return output_bboxes.view(-1, 4), iou_outputs.detach().view(-1)

    def predict(self, backbone_feats: Tensor, data_samples: SampleList,
                init_bbox: Tensor, sample_center: Tensor,
                scale_factor: float) -> Tensor:
        """Refine the target bounding box.

        Args:
            init_bbox (Tensor): of shape (4, ) or (1, 4) in [cx, cy, w, h]
                formmat.
            backbone_feats (tuple(Tensor)): of shape (1, c, h, w)
            sample_center (Tensor): The center of the cropped
                sample on the original image. It's in [x, y] format.
            scale_factor (float): The size ratio of the cropped patch to the
                resized image.

        Returns:
            Tensor: The refined target bbox in [cx, cy, w, h] format.
        """
        init_bbox = init_bbox.squeeze()
        sample_center = sample_center.squeeze()
        assert sample_center.dim() == 1

        iou_features = self(backbone_feats)
        return self.predict_by_feat(iou_features, init_bbox, sample_center,
                                    scale_factor)

    def predict_by_feat(self, iou_features: Tensor, init_bbox: Tensor,
                        sample_center: Tensor, scale_factor: Tensor) -> Tensor:
        """Refine the target bounding box.

        Args:
            init_bbox (Tensor): The init target bbox.
            iou_features (Tensor): The features for IoU prefiction.
            sample_center (Tensor): The coordinate of the sample center based
                on the original image.
            scale_factor (float): The size ratio of the cropped patch to the
                resized image.

        Returns:
            Tensor: The refined target bbox in [cx, cy, w, h] format.
        """

        # Generate some random initial boxes based on the `init_bbox`
        init_bbox = init_bbox.view(1, 4)
        init_bboxes = init_bbox.clone()
        if self.bbox_cfg['num_init_random_boxes'] > 0:
            square_box_sz = init_bbox[0, 2:].prod().sqrt().item()
            rand_factor = square_box_sz * torch.cat([
                self.bbox_cfg['box_jitter_pos'] * torch.ones(2),
                self.bbox_cfg['box_jitter_sz'] * torch.ones(2)
            ])
            min_edge_size = init_bbox[0, 2:].min() / 3
            rand_bboxes_jitter = (torch.rand(
                self.bbox_cfg['num_init_random_boxes'], 4) - 0.5) * rand_factor
            rand_bboxes_jitter = rand_bboxes_jitter.to(init_bbox.device)
            new_size = (init_bbox[:, 2:] +
                        rand_bboxes_jitter[:, 2:]).clamp(min_edge_size)
            new_center = init_bbox[:, :2] + rand_bboxes_jitter[:, :2]
            init_bboxes = torch.cat([new_center, new_size], 1)
            init_bboxes = torch.cat([init_bbox, init_bboxes])

        # Optimize the boxes
        out_bboxes, out_iou = self.optimize_bboxes(iou_features, init_bboxes)

        return self._bbox_post_process(out_bboxes, out_iou, sample_center,
                                       scale_factor)

    def _bbox_post_process(self, out_bboxes: Tensor, out_ious: Tensor,
                           sample_center: Tensor,
                           scale_factor: float) -> Tensor:
        """The post process about bbox.

        Args:
            out_bboxes (Tensor): The several optimized bboxes.
            out_ious (Tensor): The IoUs about the optimized bboxes.
            sample_center (Tensor): The coordinate of the sample center based
                on the original image.
            scale_factor (float): The size ratio of the cropped patch to the
                resized image.

        Returns:
            Tensor: The refined target bbox in [cx, cy, w, h] format.
        """
        # Remove weird boxes according to the ratio of aspect
        out_bboxes[:, 2:].clamp_(1)
        aspect_ratio = out_bboxes[:, 2] / out_bboxes[:, 3]
        keep_ind = (aspect_ratio < self.bbox_cfg['max_aspect_ratio']) * (
            aspect_ratio > 1 / self.bbox_cfg['max_aspect_ratio'])
        out_bboxes = out_bboxes[keep_ind, :]
        out_ious = out_ious[keep_ind]

        # If no box found
        if out_bboxes.shape[0] == 0:
            return None

        # Predict box
        k = self.bbox_cfg['iounet_topk']
        topk = min(k, out_bboxes.shape[0])
        _, inds = torch.topk(out_ious, topk)
        # in [x,y,w,h] format
        predicted_box = out_bboxes[inds, :].mean(0)

        # Convert the bbox of the cropped sample to that of original image.
        # TODO: this postprocess about mapping back can be moved to other place
        new_bbox_center = predicted_box[:2] + predicted_box[2:] / 2
        new_bbox_center = (new_bbox_center - self.test_cfg['img_sample_size'] /
                           2) * scale_factor + sample_center
        new_target_size = predicted_box[2:] * scale_factor

        return torch.cat([new_bbox_center, new_target_size], dim=-1)

    def _gauss_density_centered(self, x: Tensor, sigma: Tensor) -> Tensor:
        """Evaluate the probability density of a Gaussian centered at zero.

        args:
            x (Tensor): of (num_smples, 4) shape.
            sigma (Tensor): Standard deviations with (1, 4, 2) shape.
        """

        return torch.exp(-0.5 * (x / sigma)**2) / (
            math.sqrt(2 * math.pi) * sigma)

    def _gmm_density_centered(self, x: Tensor, sigma: Tensor) -> Tensor:
        """Evaluate the probability density of a GMM centered at zero.

        args:
            x(Tensor): of (num_smples, 4) shape.
            sigma (Tensor): Tensor of standard deviations with (1, 4, 2) shape.
        """
        if x.dim() == sigma.dim() - 1:
            x = x[..., None]
        elif not (x.dim() == sigma.dim() and x.shape[-1] == 1):
            raise ValueError('Last dimension must be the gmm sigmas.')

        # ``product`` along feature dim of ``bbox```, ``mean``` along component
        # dim of ``sigma``.
        return self._gauss_density_centered(x, sigma).prod(-2).mean(-1)

    def _sample_gmm_centered(self,
                             sigma: Tensor,
                             num_samples: int = 1) -> Tuple[Tensor, Tensor]:
        """Sample from a GMM distribution centered at zero.

        Args:
            sigma (Tensor): Standard deviations of bbox coordinates with
                [4, 2] shape.
            num_samples (int, optional): The number of samples.

        Returns:
            x_centered (Tensor): of shape (num_samples, num_dims)
            prob_density (Tensor): of shape (num_samples, )
        """
        num_components = sigma.shape[-1]
        num_dims = sigma.shape[-2]

        sigma = sigma.reshape(1, num_dims, num_components)

        # Sampling component index
        k = torch.randint(
            num_components, size=(num_samples, ), dtype=torch.int64)
        sigma_samples = sigma[0, :, k].t()

        x_centered = sigma_samples * torch.randn(num_samples, num_dims).to(
            sigma_samples.device)
        prob_density = self._gmm_density_centered(x_centered, sigma)

        return x_centered, prob_density

    def get_targets(self, bbox: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Generate the training targets for search images.

        Args:
            bbox (Tensor): The bbox of (N, 4) shape in [x, y, w, h] format.

        Returns:
            Tuple[Tensor, Tensor, Tensor]:
                ``proposals``: proposals with [num_samples, 4] shape.
                ``proposal_density``: proposal density with [num_samples, ]
                    shape.
                ``gt_density``: groundtruth density with [num_samples, ] shape.
        """
        bbox = bbox.clone().reshape(-1, 4)
        bbox_wh = bbox[:, 2:]

        if not hasattr(self, 'proposals_sigma'):
            center_sigma = torch.tensor(
                [s[0] for s in self.train_cfg['proposals_sigma']])
            size_sigma = torch.tensor(
                [s[1] for s in self.train_cfg['proposals_sigma']])
            # of shape (4, len(train_cfg['proposal_sigma']))
            self.proposals_sigma = torch.stack(
                (center_sigma, center_sigma, size_sigma, size_sigma),
                dim=0).to(bbox.device)

        if not hasattr(self, 'gt_bboxes_sigma'):
            # of shape (1, 4)
            self.gt_bboxes_sigma = torch.tensor(
                (self.train_cfg['gt_bboxes_sigma'][0],
                 self.train_cfg['gt_bboxes_sigma'][0],
                 self.train_cfg['gt_bboxes_sigma'][1],
                 self.train_cfg['gt_bboxes_sigma'][1]),
                dtype=torch.float32,
                device=bbox.device).reshape(-1, 4)

        # Sample boxes
        proposals_rel_centered, proposal_density = self._sample_gmm_centered(
            self.proposals_sigma,
            self.train_cfg['num_samples'] * bbox.shape[0])

        # Add mean and map back
        # of shape (num_seq*bs, 4)
        mean_box_rel = bbox_xywh_to_rel_cxcywh(bbox, bbox_wh)
        # the first num_samples elements along zero dim are
        # from the same image
        # of shape (num_samples*bbox.shape[0] , bbox.shape[-1])
        mean_box_rel = mean_box_rel.unsqueeze(1).expand(
            -1, self.train_cfg['num_samples'],
            -1).reshape(-1, mean_box_rel.shape[-1])
        proposals_rel = proposals_rel_centered + mean_box_rel
        # of shape (num_samples * bbox.shape[0], bbox.shape[-1])
        bbox_wh = bbox_wh.unsqueeze(1).expand(-1,
                                              self.train_cfg['num_samples'],
                                              -1).reshape(
                                                  -1, bbox_wh.shape[-1])
        proposals = bbox_rel_cxcywh_to_xywh(proposals_rel, bbox_wh)

        # of shape (num_samples, )
        gt_density = self._gauss_density_centered(
            proposals_rel_centered, self.gt_bboxes_sigma).prod(-1)

        if self.train_cfg['add_first_bbox']:
            proposals = torch.cat((bbox, proposals))
            proposal_density = torch.cat(
                (torch.tensor([-1.]), proposal_density))
            gt_density = torch.cat((torch.tensor([1.]), gt_density))

        return proposals, proposal_density, gt_density

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
        batch_size = len(batch_data_samples)
        batch_gt_bboxes = []
        batch_img_metas = []
        batch_search_gt_bboxes = []

        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_bboxes.append(data_sample.gt_instances['bboxes'])
            batch_search_gt_bboxes.append(
                data_sample.search_gt_instances['bboxes'])

        # Extract the first train sample in each sequence
        template_feats = [feat[:batch_size, ...] for feat in template_feats]
        batch_gt_bboxes = torch.stack(batch_gt_bboxes, dim=1)

        # Get modulation vector
        modulations = self.get_modulation(template_feats, batch_gt_bboxes[0])

        iou_feats = self(search_feats)
        num_search_imgs_per_seq = batch_data_samples[0].search_gt_instances[
            'bboxes'].shape[0]
        # (num_seq*bs, c). The first `bs` tensors along
        # zero-dim are from different images of a batch.
        modulations = [
            feat.repeat(num_search_imgs_per_seq, 1, 1, 1)
            for feat in modulations
        ]

        return self.loss_by_feat(modulations, iou_feats, batch_gt_bboxes,
                                 batch_search_gt_bboxes)

    def loss_by_feat(self, modulations: Tuple[Tensor],
                     iou_feats: Tuple[Tensor], batch_gt_bboxes: Tensor,
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
        batch_search_gt_bboxes = torch.stack(
            batch_search_gt_bboxes, dim=1).view(-1, 4)
        batch_search_gt_bboxes_xywh = bbox_xyxy_to_x1y1wh(
            batch_search_gt_bboxes)
        (proposals, proposals_density, search_gt_bboxes_density
         ) = self.get_targets(batch_search_gt_bboxes_xywh)

        proposals = proposals.view(-1, self.train_cfg['num_samples'],
                                   proposals.shape[-1])
        proposals_density = proposals_density.view(
            -1, self.train_cfg['num_samples'])
        search_gt_bboxes_density = search_gt_bboxes_density.view(
            -1, self.train_cfg['num_samples'])
        pred_iou = self.predict_iou(modulations, iou_feats, proposals)

        loss_bbox = self.loss_bbox(
            pred_iou,
            sample_density=proposals_density,
            gt_density=search_gt_bboxes_density,
            mc_dim=1) * self.train_cfg['loss_weights']['bbox']

        return dict(loss_bbox=loss_bbox)
