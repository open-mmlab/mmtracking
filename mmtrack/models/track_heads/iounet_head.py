# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn.bricks import ConvModule
from mmdet.core.bbox.transforms import bbox_cxcywh_to_xyxy
from mmdet.models import HEADS

from mmtrack.core.bbox import (bbox_cxcywh_to_x1y1wh, bbox_rect_to_rel,
                               bbox_rel_to_rect)
from ..utils.PreciseRoIPooling.pytorch.prroi_pool import PrRoIPool2D


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
                 in_planes,
                 out_planes,
                 input_size,
                 bias=True,
                 batch_norm=True,
                 relu=True):
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


@HEADS.register_module()
class IouNetHead(nn.Module):
    """Module for IoU prediction.

    Refer to the ATOM paper for a detailed illustration of the architecture.
    `ATOM: <https://arxiv.org/abs/1811.07628>`_.

    Args:
        input_dim (tuple(int)): Feature dimensionality of the two input
            backbone layers.
        pred_input_dim (tuple(int)): Input dimensionality of the the
            prediction network.
        pred_inter_dim (tuple(int)): Intermediate dimensionality in the
            prediction network.
        bbox_cfg (dict, optional): The configuration of bbox refinement.
            Defaults to None.
        train_cfg (dict, optional): The configuration of training.
            Defaults to None.
        loss_bbox (dict, optional): The configuration of loss.
            Defaults to None.
    """

    def __init__(self,
                 input_dim=(128, 256),
                 pred_input_dim=(256, 256),
                 pred_inter_dim=(256, 256),
                 bbox_cfg=None,
                 train_cfg=None,
                 loss_bbox=None,
                 **kwargs):
        super().__init__()
        self.bbox_cfg = bbox_cfg

        def conv_module(in_planes, out_planes, kernel_size=3, padding=1):
            # The module's pipeline: Conv -> BN -> ReLU.
            return ConvModule(
                in_channels=in_planes,
                out_channels=out_planes,
                kernel_size=kernel_size,
                padding=padding,
                bias=True,
                norm_cfg=dict(type='BN', requires_grad=True),
                act_cfg=dict(type='ReLU'),
                inplace=True)

        # `*_temp` denotes template branch, `*_search` denotes search branch
        self.conv3_temp = conv_module(input_dim[0], 128)
        self.roi3_temp = PrRoIPool2D(3, 3, 1 / 8)
        self.fc3_temp = conv_module(128, 256, padding=0)
        self.fc34_3_temp = conv_module(
            256 + 256, pred_input_dim[0], kernel_size=1, padding=0)

        self.conv4_temp = conv_module(input_dim[1], 256)
        self.roi4_temp = PrRoIPool2D(1, 1, 1 / 16)
        self.fc34_4_temp = conv_module(
            256 + 256, pred_input_dim[1], kernel_size=1, padding=0)

        self.conv3_search = nn.Sequential(
            conv_module(input_dim[0], 256),
            conv_module(256, pred_input_dim[0]))
        self.roi3_search = PrRoIPool2D(5, 5, 1 / 8)
        self.fc3_search = LinearBlock(pred_input_dim[0], pred_inter_dim[0], 5)

        self.conv4_search = nn.Sequential(
            conv_module(input_dim[1], 256),
            conv_module(256, pred_input_dim[1]))
        self.roi4_search = PrRoIPool2D(3, 3, 1 / 16)
        self.fc4_search = LinearBlock(pred_input_dim[1], pred_inter_dim[1], 3)

        self.iou_predictor = nn.Linear(
            pred_inter_dim[0] + pred_inter_dim[1], 1, bias=True)

    def init_weights(self):
        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(
                    m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.uniform_()
                m.bias.data.zero_()

    def predict_iou(self, modulations, feats, proposals):
        """Predicts IoU for the give proposals.

        args:
            modulations (tuple(Tensor)): contains the features from two layers.
                The inner tensors are of shape (bs, c, 1, 1)
            feats (tuple(Tensor)):  IoU features for test images.
                The inner tensors are of shape (bs, c, h, w).
            proposals:  Proposal boxes for which the IoU will be predicted
                (bs, num_proposals, 4).
        """

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

    def get_modulation(self, feats, bboxes):
        """Get modulation vectors for the targets.

        args:
            feats (tuple(Tensor)): Backbone features from template branch.
                It's of shape (batch, feature_dim, H, W).
            bboxes: Target boxes (x1, y1, x2, y2) in image coords in the
                template branch. It's of shape (batch, 4).
        """

        # Add batch_index to rois
        batch_size = bboxes.shape[0]
        batch_index = torch.arange(
            batch_size, dtype=torch.float32).reshape(-1, 1).to(bboxes.device)
        roi = torch.cat((batch_index, bboxes), dim=1)

        # Perform conv and prpool on the feature maps from the backbone
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

    def get_iou_feat(self, feats):
        """Get IoU prediction features from a 4 or 5 dimensional backbone
        input."""
        feats = [
            f.reshape(-1, *f.shape[-3:]) if f.dim() == 5 else f for f in feats
        ]
        feat3_search, feat4_search = feats
        c3_search = self.conv3_search(feat3_search)
        c4_search = self.conv4_search(feat4_search)

        return c3_search, c4_search

    def init_iou_net(self, iou_backbone_feats, bboxes):
        """Initialize the IoUNet with feature are from the 'layer2' and
        'layer3' of backbone.

        Args:
            iou_backbone_feats (tuple(Tensor)): _description_
            bboxes (Tensor): of shape (4, ) in [cx, cy, w, h] format.
        """
        bboxes = bbox_cxcywh_to_xyxy(bboxes.view(-1, 4))
        # Get modulation vector
        with torch.no_grad():
            self.iou_modulation = self.get_modulation(iou_backbone_feats,
                                                      bboxes)

    def optimize_bboxes(self, iou_features, init_bboxes):
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
        output_bboxes_rel = bbox_rect_to_rel(output_bboxes, bboxes_sz_norm)

        with torch.set_grad_enabled(True):
            for _ in range(self.bbox_cfg['box_refine_iter']):
                # Forward
                bboxes_init_rel = output_bboxes_rel.clone().detach()
                bboxes_init_rel.requires_grad = True

                bboxes_init = bbox_rel_to_rect(bboxes_init_rel, bboxes_sz_norm)
                iou_outputs = self.predict_iou(self.iou_modulation,
                                               iou_features, bboxes_init)
                # Backward
                iou_outputs.backward(gradient=torch.ones_like(iou_outputs))

                # Update bboxes
                output_bboxes_rel = bboxes_init_rel + (
                    step_length * bboxes_init_rel.grad)
                output_bboxes_rel.detach_()

                step_length *= self.bbox_cfg['box_refine_step_decay']

        output_bboxes = bbox_rel_to_rect(output_bboxes_rel, bboxes_sz_norm)

        return output_bboxes.view(-1, 4), iou_outputs.detach().view(-1)

    def refine_target_bbox(self, init_bbox, backbone_feats, sample_center,
                           sample_scale, sample_size):
        """Run the ATOM IoUNet to refine the target bounding box.

        Args:
            init_bbox (Tensor): of shape (4, ) in [cx, cy, w, h] formmat.
            backbone_feats (tuple(Tensor)): of shape (1, c, h, w)
            sample_center (Tensor): The center of the cropped
                sample on the original image. It's of shape (1,2) in [x, y]
                format.
            sample_scale (Tensor): The scale of the cropped sample. It's of
                shape (1,).
            sample_size (Tensor): The scale of the cropped sample. It's of
                shape (2,) in [h, w] format.

        Returns:
            Tensor: new target bbox in [cx, cy, w, h] format.
        """
        with torch.no_grad():
            iou_features = self.get_iou_feat(backbone_feats)

        # Generate some random initial boxes based on the `init_bbox`
        init_bboxes = init_bbox.view(1, 4).clone()
        if self.bbox_cfg['num_init_random_boxes'] > 0:
            square_box_sz = init_bbox[2:].prod().sqrt().item()
            rand_factor = square_box_sz * torch.cat([
                self.bbox_cfg['box_jitter_pos'] * torch.ones(2),
                self.bbox_cfg['box_jitter_sz'] * torch.ones(2)
            ])
            mini_edge_size = init_bbox[2:].min() / 3
            rand_bboxes_shift = (torch.rand(
                self.bbox_cfg['num_init_random_boxes'], 4) - 0.5) * rand_factor
            rand_bboxes_shift = rand_bboxes_shift.to(init_bbox.device)
            new_size = (init_bbox[2:] +
                        rand_bboxes_shift[:, 2:]).clamp(mini_edge_size)
            new_center = init_bbox[:2] + rand_bboxes_shift[:, :2]
            init_bboxes = torch.cat([new_center, new_size], 1)
            init_bboxes = torch.cat([init_bbox.view(1, 4), init_bboxes])

        # Optimize the boxes
        out_bboxes, out_iou = self.optimize_bboxes(iou_features, init_bboxes)

        # Remove weird boxes according to the ratio of aspect
        out_bboxes[:, 2:].clamp_(1)
        aspect_ratio = out_bboxes[:, 2] / out_bboxes[:, 3]
        keep_ind = (aspect_ratio < self.bbox_cfg['max_aspect_ratio']) * (
            aspect_ratio > 1 / self.bbox_cfg['max_aspect_ratio'])
        out_bboxes = out_bboxes[keep_ind, :]
        out_iou = out_iou[keep_ind]

        # If no box found
        if out_bboxes.shape[0] == 0:
            return None

        # Predict box
        k = self.bbox_cfg.get('iounet_topk', 5)
        topk = min(k, out_bboxes.shape[0])
        _, inds = torch.topk(out_iou, topk)
        # in [x,y,w,h] format
        predicted_box = out_bboxes[inds, :].mean(0)

        # Convert the bbox of the cropped sample to that of original image.
        # TODO: this postprocess about mapping back can be moved to other place
        new_bbox_center = predicted_box[:2] + predicted_box[2:] / 2
        new_bbox_center = (new_bbox_center - sample_size /
                           2) * sample_scale[0] + sample_center[0]
        new_target_size = predicted_box[2:] * sample_scale[0]

        return torch.cat([new_bbox_center, new_target_size], dim=-1)
