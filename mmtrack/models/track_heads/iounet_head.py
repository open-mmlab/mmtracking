# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmdet.models import HEADS

from mmtrack.core.utils import TensorList
from ..utils.PreciseRoIPooling.pytorch.prroi_pool import PrRoIPool2D


class LinearBlock(nn.Module):

    def __init__(self,
                 in_planes,
                 out_planes,
                 input_sz,
                 bias=True,
                 batch_norm=True,
                 relu=True):
        super().__init__()
        self.linear = nn.Linear(
            in_planes * input_sz * input_sz, out_planes, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes) if batch_norm else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.linear(x.reshape(x.shape[0], -1))
        if self.bn is not None:
            x = self.bn(x.reshape(x.shape[0], x.shape[1], 1, 1))
        if self.relu is not None:
            x = self.relu(x)
        return x.reshape(x.shape[0], -1)


def rel_to_rect(bb, sz_norm=None):
    """Inverts the effect of rect_to_rel.

    See above.
    """

    sz = torch.exp(bb[..., 2:])
    if sz_norm is None:
        c = bb[..., :2] * sz
    else:
        c = bb[..., :2] * sz_norm
    tl = c - 0.5 * sz
    return torch.cat((tl, sz), dim=-1)


def rect_to_rel(bb, sz_norm=None):
    """Convert standard rectangular parametrization of the bounding box.

    [x, y, w, h] to relative parametrization [cx/sw, cy/sh, log(w), log(h)],
    where [cx, cy] is the center coordinate.
    args:
        bb  -  N x 4 tensor of boxes.
        sz_norm  -  [N] x 2 tensor of value of [sw, sh] (optional).
        sw=w and sh=h if not given.
    """

    c = bb[..., :2] + 0.5 * bb[..., 2:]
    if sz_norm is None:
        c_rel = c / bb[..., 2:]
    else:
        c_rel = c / sz_norm
    sz_rel = torch.log(bb[..., 2:])
    return torch.cat((c_rel, sz_rel), dim=-1)


@HEADS.register_module()
class IouNetHead(nn.Module):
    """Network module for IoU prediction.

    Refer to the ATOM paper for an
        illustration of the architecture.
        It uses two backbone feature layers as input.
    args:
        input_dim:  Feature dimensionality of the two input backbone layers.
        pred_input_dim:  Dimensionality input the the prediction network.
        pred_inter_dim:  Intermediate dimensionality in the prediction network.
    """

    def __init__(self,
                 input_dim=(128, 256),
                 pred_input_dim=(256, 256),
                 pred_inter_dim=(256, 256),
                 img_sample_sz=352,
                 bbox_cfg=None,
                 **kwargs):
        super().__init__()

        self.img_sample_sz = img_sample_sz
        self.bbox_cfg = bbox_cfg

        def conv(in_planes,
                 out_planes,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1):
            return nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    out_planes,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    bias=True), nn.BatchNorm2d(out_planes),
                nn.ReLU(inplace=True))

        # _r for reference, _t for test
        self.conv3_1r = conv(input_dim[0], 128, kernel_size=3, stride=1)
        self.conv3_1t = conv(input_dim[0], 256, kernel_size=3, stride=1)

        self.conv3_2t = conv(256, pred_input_dim[0], kernel_size=3, stride=1)

        self.prroi_pool3r = PrRoIPool2D(3, 3, 1 / 8)
        self.prroi_pool3t = PrRoIPool2D(5, 5, 1 / 8)

        self.fc3_1r = conv(128, 256, kernel_size=3, stride=1, padding=0)

        self.conv4_1r = conv(input_dim[1], 256, kernel_size=3, stride=1)
        self.conv4_1t = conv(input_dim[1], 256, kernel_size=3, stride=1)

        self.conv4_2t = conv(256, pred_input_dim[1], kernel_size=3, stride=1)

        self.prroi_pool4r = PrRoIPool2D(1, 1, 1 / 16)
        self.prroi_pool4t = PrRoIPool2D(3, 3, 1 / 16)

        self.fc34_3r = conv(
            256 + 256, pred_input_dim[0], kernel_size=1, stride=1, padding=0)
        self.fc34_4r = conv(
            256 + 256, pred_input_dim[1], kernel_size=1, stride=1, padding=0)

        self.fc3_rt = LinearBlock(pred_input_dim[0], pred_inter_dim[0], 5)
        self.fc4_rt = LinearBlock(pred_input_dim[1], pred_inter_dim[1], 3)

        self.iou_predictor = nn.Linear(
            pred_inter_dim[0] + pred_inter_dim[1], 1, bias=True)

        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(
                    m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                # In earlier versions batch norm parameters was initialized
                # with default initialization,
                # which changed in pytorch 1.2. In 1.1 and earlier the weight
                # was set to U(0,1).
                # So we use the same initialization here.
                # m.weight.data.fill_(1)
                m.weight.data.uniform_()
                m.bias.data.zero_()

    def forward(self, feat1, feat2, bb1, proposals2):
        """Runs the ATOM IoUNet during training operation.

        This forward pass is mainly used for training. Call the individual
            functions during tracking instead.
        args:
            feat1:  Features from the reference frames (4 or 5 dims).
            feat2:  Features from the test frames (4 or 5 dims).
            bb1:  Target boxes (x,y,w,h) in image coords in the reference
                samples. Dims (images, sequences, 4).
            proposals2:  Proposal boxes for which the IoU will be predicted
                (images, sequences, num_proposals, 4).
        """

        assert bb1.dim() == 3
        assert proposals2.dim() == 4

        num_images = proposals2.shape[0]
        num_sequences = proposals2.shape[1]

        # Extract first train sample
        feat1 = [
            f[0, ...] if f.dim() == 5 else f.reshape(
                -1, num_sequences, *f.shape[-3:])[0, ...] for f in feat1
        ]
        bb1 = bb1[0, ...]

        # Get modulation vector
        modulation = self.get_modulation(feat1, bb1)

        iou_feat = self.get_iou_feat(feat2)

        modulation = [
            f.reshape(1, num_sequences,
                      -1).repeat(num_images, 1,
                                 1).reshape(num_sequences * num_images, -1)
            for f in modulation
        ]

        proposals2 = proposals2.reshape(num_sequences * num_images, -1, 4)
        pred_iou = self.predict_iou(modulation, iou_feat, proposals2)
        return pred_iou.reshape(num_images, num_sequences, -1)

    def predict_iou(self, modulation, feat, proposals):
        """Predicts IoU for the give proposals.

        args:
            modulation:  Modulation vectors for the targets.
                Dims (batch, feature_dim).
            feat:  IoU features (from get_iou_feat) for test images.
                Dims (batch, feature_dim, H, W).
            proposals:  Proposal boxes for which the IoU will be predicted
                (batch, num_proposals, 4).
        """

        fc34_3_r, fc34_4_r = modulation
        c3_t, c4_t = feat

        batch_size = c3_t.size()[0]

        # Modulation
        c3_t_att = c3_t * fc34_3_r.reshape(batch_size, -1, 1, 1)
        c4_t_att = c4_t * fc34_4_r.reshape(batch_size, -1, 1, 1)

        # Add batch_index to rois
        batch_index = torch.arange(
            batch_size, dtype=torch.float32).reshape(-1, 1).to(c3_t.device)

        # Push the different rois for the same image along the batch dimension
        num_proposals_per_batch = proposals.shape[1]

        # input proposals2 is in format xywh, convert it to x0y0x1y1 format
        proposals_xyxy = torch.cat(
            (proposals[:, :,
                       0:2], proposals[:, :, 0:2] + proposals[:, :, 2:4]),
            dim=2)

        # Add batch index
        roi2 = torch.cat((batch_index.reshape(batch_size, -1, 1).expand(
            -1, num_proposals_per_batch, -1), proposals_xyxy),
                         dim=2)
        roi2 = roi2.reshape(-1, 5).to(proposals_xyxy.device)

        roi3t = self.prroi_pool3t(c3_t_att, roi2)
        roi4t = self.prroi_pool4t(c4_t_att, roi2)

        fc3_rt = self.fc3_rt(roi3t)
        fc4_rt = self.fc4_rt(roi4t)

        fc34_rt_cat = torch.cat((fc3_rt, fc4_rt), dim=1)

        iou_pred = self.iou_predictor(fc34_rt_cat).reshape(
            batch_size, num_proposals_per_batch)

        return iou_pred

    def get_modulation(self, feat, bb):
        """Get modulation vectors for the targets.

        args:
            feat: Backbone features from reference images. Dims
                (batch, feature_dim, H, W).
            bb:  Target boxes (x,y,w,h) in image coords in the reference
                samples. Dims (batch, 4).
        """

        feat3_r, feat4_r = feat

        c3_r = self.conv3_1r(feat3_r)

        # Add batch_index to rois
        batch_size = bb.shape[0]
        batch_index = torch.arange(
            batch_size, dtype=torch.float32).reshape(-1, 1).to(bb.device)

        # input bb is in format xywh, convert it to x0y0x1y1 format
        bb = bb.clone()
        bb[:, 2:4] = bb[:, 0:2] + bb[:, 2:4]
        roi1 = torch.cat((batch_index, bb), dim=1)

        roi3r = self.prroi_pool3r(c3_r, roi1)

        c4_r = self.conv4_1r(feat4_r)
        roi4r = self.prroi_pool4r(c4_r, roi1)

        fc3_r = self.fc3_1r(roi3r)

        # Concatenate from block 3 and 4
        fc34_r = torch.cat((fc3_r, roi4r), dim=1)

        fc34_3_r = self.fc34_3r(fc34_r)
        fc34_4_r = self.fc34_4r(fc34_r)

        return fc34_3_r, fc34_4_r

    def get_iou_feat(self, feat2):
        """Get IoU prediction features from a 4 or 5 dimensional backbone
        input."""
        feat2 = [
            f.reshape(-1, *f.shape[-3:]) if f.dim() == 5 else f for f in feat2
        ]
        feat3_t, feat4_t = feat2
        c3_t = self.conv3_2t(self.conv3_1t(feat3_t))
        c4_t = self.conv4_2t(self.conv4_1t(feat4_t))

        return c3_t, c4_t

    def init_iou_net(self, iou_backbone_feat, bboxes):
        # iou_backbone feature are from the 'layer2' and 'layer3' of backbone
        # by default.

        # Setup IoU net and objective
        for p in self.parameters():
            p.requires_grad = False

        target_boxes = TensorList()
        target_boxes.append(bboxes)
        target_boxes = torch.cat(target_boxes.view(1, 4), 0)

        # Remove other augmentations such as rotation
        iou_backbone_feat = TensorList(
            [x[:target_boxes.shape[0], ...] for x in iou_backbone_feat])

        # Get modulation vector
        self.iou_modulation = self.get_iou_modulation(iou_backbone_feat,
                                                      target_boxes)
        if torch.is_tensor(self.iou_modulation[0]):
            self.iou_modulation = TensorList(
                [x.detach().mean(0) for x in self.iou_modulation])

    def get_iou_modulation(self, iou_backbone_feat, target_boxes):
        with torch.no_grad():
            return self.get_modulation(iou_backbone_feat, target_boxes)

    def generate_bbox(self, pos, sz, sample_pos, sample_scale):
        """All inputs in original image coordinates.

        Generates a box in the cropped image sample reference frame, in the
        format used by the IoUNet.
        """
        box_center = (pos - sample_pos) / sample_scale + (self.img_sample_sz -
                                                          1) / 2
        box_sz = sz / sample_scale
        target_ul = box_center - (box_sz - 1) / 2
        return torch.cat([target_ul.flip((0, )), box_sz.flip((0, ))])

    def optimize_bboxes(self, iou_features, init_boxes):
        """Optimize iounet boxes with the relative parametrization ised in
        PrDiMP."""
        output_boxes = init_boxes.view(1, -1, 4).to(iou_features[0].device)
        step_length = self.bbox_cfg['box_refinement_step_length']
        if isinstance(step_length, (tuple, list)):
            step_length = torch.Tensor([
                step_length[0], step_length[0], step_length[1], step_length[1]
            ]).to(iou_features[0].device).view(1, 1, 4)

        sz_norm = output_boxes[:, :1, 2:].clone()
        output_boxes_rel = rect_to_rel(output_boxes, sz_norm)
        for i_ in range(self.bbox_cfg['box_refinement_iter']):
            # forward pass
            bb_init_rel = output_boxes_rel.clone().detach()
            bb_init_rel.requires_grad = True

            bb_init = rel_to_rect(bb_init_rel, sz_norm)
            outputs = self.predict_iou(self.iou_modulation, iou_features,
                                       bb_init)

            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]

            outputs.backward(gradient=torch.ones_like(outputs))

            # Update proposal
            output_boxes_rel = bb_init_rel + step_length * bb_init_rel.grad
            output_boxes_rel.detach_()

            step_length *= self.bbox_cfg['box_refinement_step_decay']

        output_boxes = rel_to_rect(output_boxes_rel, sz_norm)

        return output_boxes.view(-1, 4).cpu(), outputs.detach().view(-1).cpu()

    def refine_target_box(self, init_bbox, backbone_feat, sample_pos,
                          sample_scale, scale_ind):
        """Run the ATOM IoUNet to refine the target bounding box."""
        # Extract features from the relevant scale
        # backbone_feat is a tuple
        with torch.no_grad():
            iou_features = self.get_iou_feat(backbone_feat)
        iou_features = TensorList(
            [x[scale_ind:scale_ind + 1, ...] for x in iou_features])

        # Generate random initial boxes
        init_bboxes = init_bbox.view(1, 4).clone()
        if self.bbox_cfg['num_init_random_boxes'] > 0:
            square_box_sz = init_bbox[2:].prod().sqrt()
            rand_factor = square_box_sz * torch.cat([
                self.bbox_cfg['box_jitter_pos'] * torch.ones(2),
                self.bbox_cfg['box_jitter_sz'] * torch.ones(2)
            ])

            minimal_edge_size = init_bbox[2:].min() / 3
            rand_bb = (torch.rand(self.bbox_cfg['num_init_random_boxes'], 4) -
                       0.5) * rand_factor
            new_sz = (init_bbox[2:] + rand_bb[:, 2:]).clamp(minimal_edge_size)
            new_center = (init_bbox[:2] + init_bbox[2:] / 2) + rand_bb[:, :2]
            init_bboxes = torch.cat([new_center - new_sz / 2, new_sz], 1)
            init_bboxes = torch.cat([init_bbox.view(1, 4), init_bboxes])

        # Optimize the boxes
        output_boxes, output_iou = self.optimize_bboxes(
            iou_features, init_bboxes.to(backbone_feat[-1].device))

        # Remove weird boxes
        output_boxes[:, 2:].clamp_(1)
        aspect_ratio = output_boxes[:, 2] / output_boxes[:, 3]
        keep_ind = (aspect_ratio < self.bbox_cfg['maximal_aspect_ratio']) * (
            aspect_ratio > 1 / self.bbox_cfg['maximal_aspect_ratio'])
        output_boxes = output_boxes[keep_ind, :]
        output_iou = output_iou[keep_ind]

        # If no box found
        if output_boxes.shape[0] == 0:
            return None, None

        # Predict box
        k = self.bbox_cfg.get('iounet_k', 5)
        topk = min(k, output_boxes.shape[0])
        _, inds = torch.topk(output_iou, topk)
        # in [x,y,w,h] format
        predicted_box = output_boxes[inds, :].mean(0)

        # Get new position and size
        new_pos = predicted_box[:2] + predicted_box[2:] / 2
        new_pos = (new_pos.flip(
            (0, )) - (self.img_sample_sz - 1) / 2) * sample_scale + sample_pos
        new_target_sz = predicted_box[2:].flip((0, )) * sample_scale

        return new_pos, new_target_sz
