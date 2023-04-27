# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import torchvision.ops as ops
from mmdet.models.dense_heads import DeformableDETRHead
from mmdet.models.layers import inverse_sigmoid
from mmdet.models.utils import multi_apply
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from mmdet.utils import ConfigType, InstanceList, OptInstanceList, reduce_mean
from mmengine.structures import InstanceData
from torch import Tensor

from mmtrack.registry import MODELS
from .pos_neg_select import get_contrast_items
from .utils import (MLP, MaskHeadSmallConv, aligned_bilinear,
                    compute_locations, parse_dynamic_params)


@MODELS.register_module()
class IDOLTrackHead(DeformableDETRHead):

    def __init__(self,
                 *args,
                 loss_mask: ConfigType = dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=20.0),
                 loss_dice: ConfigType = dict(
                     type='DiceLoss',
                     use_sigmoid=True,
                     activate=True,
                     naive_dice=True,
                     loss_weight=1.0),
                 loss_track: ConfigType = dict(
                     type='MultiPosCrossEntropyLoss', loss_weight=0.25),
                 loss_track_aux: ConfigType = dict(
                     type='L2Loss',
                     sample_ratio=3,
                     margin=0.3,
                     loss_weight=1.0,
                     hard_mining=True),
                 enable_reid: bool = True,
                 rel_coord: bool = True,
                 inference_select_thres: float = 0.1,
                 **kwargs) -> None:

        super().__init__(*args, **kwargs)

        self.enable_reid = enable_reid
        self.rel_coord = rel_coord
        self.inference_select_thres = inference_select_thres
        embed_dims = self.transformer.embed_dims

        self.in_channels = embed_dims // 32
        self.dynamic_mask_channels = 8
        self.controller_layers = 3
        self.max_insts_num = 100
        self.mask_out_stride = 4
        self.up_rate = 8 // self.mask_out_stride

        # dynamic_mask_head params
        weight_nums, bias_nums = [], []
        for layer in range(self.controller_layers):
            if layer == 0:
                if self.rel_coord:
                    weight_nums.append(
                        (self.in_channels + 2) * self.dynamic_mask_channels)
                else:
                    weight_nums.append(self.in_channels *
                                       self.dynamic_mask_channels)
                bias_nums.append(self.dynamic_mask_channels)
            elif layer == self.controller_layers - 1:
                weight_nums.append(self.dynamic_mask_channels * 1)
                bias_nums.append(1)
            else:
                weight_nums.append(self.dynamic_mask_channels *
                                   self.dynamic_mask_channels)
                bias_nums.append(self.dynamic_mask_channels)

        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)

        self.controller = MLP(embed_dims, embed_dims, self.num_gen_params, 3)
        self.simple_conv = MaskHeadSmallConv(embed_dims, None, embed_dims)

        if self.enable_reid:
            self.reid_embed_head = MLP(embed_dims, embed_dims, embed_dims, 3)

        self.loss_mask = MODELS.build(loss_mask)
        self.loss_dice = MODELS.build(loss_dice)
        self.loss_track = MODELS.build(loss_track)
        self.loss_track_aux = MODELS.build(loss_track_aux)

    def forward(self, x: Tuple[Tensor],
                batch_img_metas: List[dict]) -> Tuple[Tensor, ...]:
        """Forward function.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
        """
        if self.training:
            return self.training_forward(x, batch_img_metas)
        else:
            return self.inference_forward(x, batch_img_metas)

    def training_forward(self, x: Tuple[Tensor],
                         batch_img_metas: List[dict]) -> Tuple[Tensor, ...]:

        batch_size = x[0].size(0)
        num_frames = len(batch_img_metas[0]['frame_id'])
        # Since ref_img exists, batch_img_metas[0]['batch_input_shape'][0]
        # means the key img.
        input_img_h, input_img_w = batch_img_metas[0]['batch_input_shape'][0]
        img_masks = x[0].new_ones((batch_size, input_img_h, input_img_w))
        for batch_id in range(batch_size // num_frames):
            img_h, img_w = batch_img_metas[batch_id]['img_shape'][0]
            img_masks[batch_id * num_frames:(batch_id + 1) *
                      num_frames, :img_h, :img_w] = 0

        mlvl_masks = []
        mlvl_positional_encodings = []
        spatial_shapes = []

        for feat in x:
            mlvl_masks.append(
                F.interpolate(img_masks[None],
                              size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_positional_encodings.append(
                self.positional_encoding(mlvl_masks[-1]))
            spatial_shapes.append((feat.shape[2], feat.shape[3]))

        query_embeds = None
        if not self.as_two_stage:
            query_embeds = self.query_embedding.weight

        # record key img info
        x_key = []
        mlvl_masks_key = []
        mlvl_positional_encodings_key = []
        # record ref img info
        x_ref = []
        mlvl_masks_ref = []
        mlvl_positional_encodings_ref = []

        key_ids = list(range(0, batch_size - 1, 2))
        ref_ids = list(range(1, batch_size, 2))

        # get key frame and ref frame infos
        for n_l in range(self.transformer.num_feature_levels):
            x_key.append(x[n_l][key_ids])
            x_ref.append(x[n_l][ref_ids])

            mlvl_masks_key.append(mlvl_masks[n_l][key_ids])
            mlvl_masks_ref.append(mlvl_masks[n_l][ref_ids])

            mlvl_positional_encodings_key.append(
                mlvl_positional_encodings[n_l][key_ids])
            mlvl_positional_encodings_ref.append(
                mlvl_positional_encodings[n_l][ref_ids])

        hs, memory, init_reference, inter_references, \
            enc_outputs_class, enc_outputs_coord = self.transformer(
                    x_key,
                    mlvl_masks_key,
                    query_embeds,
                    mlvl_positional_encodings_key,
                    reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                    cls_branches=self.cls_branches if self.as_two_stage else None  # noqa:E501
            )

        hs_ref, memory_ref, init_reference_ref, inter_references_ref, \
            enc_outputs_class_ref, enc_outputs_coord_ref = \
            self.transformer(
                x_ref,
                mlvl_masks_ref,
                query_embeds,
                mlvl_positional_encodings_ref,
                reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                cls_branches=self.cls_branches if self.as_two_stage else None  # noqa:E501
            )

        hs = hs.permute(0, 2, 1, 3)
        hs_ref = hs_ref.permute(0, 2, 1, 3)

        outputs_classes = []
        outputs_coords = []
        outputs_masks = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

            # [bs, num_quries, num_params]
            dynamic_mask_head_params = self.controller(hs[lvl])
            mask_head_params = dynamic_mask_head_params.reshape(
                1, -1, dynamic_mask_head_params.shape[-1])

            reference_points = []
            for batch_id in range(batch_size // num_frames):
                img_h, img_w = batch_img_metas[batch_id]['img_shape'][0]
                img_h = torch.as_tensor(img_h).to(reference[batch_id])
                img_w = torch.as_tensor(img_w).to(reference[batch_id])
                scale_f = torch.stack([img_w, img_h], dim=0)
                ref_cur_f = reference[batch_id].sigmoid()[..., :2] * scale_f[
                    None, :]
                reference_points.append(ref_cur_f.unsqueeze(0))

            reference_points = torch.cat(reference_points, dim=1)
            num_insts = [
                self.num_query for i in range(batch_size // num_frames)
            ]
            outputs_mask = self.mask_head(memory, spatial_shapes,
                                          reference_points, mask_head_params,
                                          num_insts)
            outputs_masks.append(outputs_mask)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        outputs_masks = torch.stack(outputs_masks)
        outputs_embeds = InstanceData(
            ref_embeds=self.reid_embed_head(hs_ref[-1]),
            key_embeds=self.reid_embed_head(hs[-1]),
            ref_cls=self.cls_branches[-1](hs_ref[-1]),
            ref_bbox=inter_references_ref[-1])
        # not support two_stage yet
        return outputs_classes, outputs_coords, \
            outputs_masks, outputs_embeds, None, None

    def inference_forward(self, x: Tuple[Tensor],
                          batch_img_metas: List[dict]) -> Tuple[Tensor, ...]:

        batch_size = x[0].size(0)

        input_img_h, input_img_w = batch_img_metas[0]['batch_input_shape']
        img_masks = x[0].new_ones((batch_size, input_img_h, input_img_w))
        for batch_id in range(batch_size):
            img_h, img_w = batch_img_metas[batch_id]['img_shape']
            img_masks[batch_id, :img_h, :img_w] = 0

        mlvl_masks = []
        mlvl_positional_encodings = []
        spatial_shapes = []

        for feat in x:
            # feat: (1, c, h, w)
            mlvl_masks.append(
                F.interpolate(img_masks[None],
                              size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_positional_encodings.append(
                self.positional_encoding(mlvl_masks[-1]))
            spatial_shapes.append((feat.shape[2], feat.shape[3]))

        query_embeds = None
        if not self.as_two_stage:
            query_embeds = self.query_embedding.weight

        hs, memory, init_reference, inter_references, \
            enc_outputs_class, enc_outputs_coord = self.transformer(
                    x,
                    mlvl_masks,
                    query_embeds,
                    mlvl_positional_encodings,
                    reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                    cls_branches=self.cls_branches if self.as_two_stage else None  # noqa:E501
            )

        hs = hs.permute(0, 2, 1, 3)

        reference = inter_references[-1 - 1]
        reference = inverse_sigmoid(reference)
        outputs_class = self.cls_branches[-1](hs[-1])
        tmp = self.reg_branches[-1](hs[-1])
        if reference.shape[-1] == 4:
            tmp += reference
        else:
            assert reference.shape[-1] == 2
            tmp[..., :2] += reference
        outputs_coord = tmp.sigmoid()

        outputs_embeds = self.reid_embed_head(hs[-1])

        dynamic_mask_head_params = self.controller(
            hs[-1])  # [bs, num_queries, num_params]

        norm_reference_points = inter_references[-2, :, :, :2]

        reference_points = []
        for batch_id in range(batch_size):
            img_h, img_w = batch_img_metas[batch_id]['img_shape']
            img_h = torch.as_tensor(img_h).to(norm_reference_points[batch_id])
            img_w = torch.as_tensor(img_w).to(norm_reference_points[batch_id])
            scale_f = torch.stack([img_w, img_h], dim=0)
            ref_cur_f = norm_reference_points[batch_id] * scale_f[None, :]
            reference_points.append(ref_cur_f.unsqueeze(0))

        reference_points = torch.cat(reference_points, dim=1)
        mask_head_params = dynamic_mask_head_params.reshape(
            1, -1, dynamic_mask_head_params.shape[-1])

        num_insts = [self.num_query for i in range(batch_size)]
        outputs_masks = self.mask_head(memory, spatial_shapes,
                                       reference_points, mask_head_params,
                                       num_insts)
        # not support two_stage yet
        return outputs_class, outputs_coord, \
            outputs_masks, outputs_embeds, None, None

    def mask_head(self, feats, spatial_shapes, reference_points,
                  mask_head_params, num_insts):

        feats = feats.transpose(0, 1)
        bs, _, c = feats.shape

        # encod_feat_l: num_layers x [bs, C, hi, wi]
        encod_feat_l = []
        spatial_indx = 0
        for lvl in range(self.transformer.num_feature_levels - 1):
            h, w = spatial_shapes[lvl]
            mem_l = feats[:, spatial_indx:spatial_indx + 1 * h * w, :].reshape(
                bs, h, w, c).permute(0, 3, 1, 2)
            encod_feat_l.append(mem_l)
            spatial_indx += 1 * h * w

        decod_feat_f = self.simple_conv(encod_feat_l, fpns=None)

        mask_logits = self.dynamic_mask_with_coords(
            decod_feat_f,
            reference_points,
            mask_head_params,
            num_insts=num_insts,
            mask_feat_stride=8,
            rel_coord=self.rel_coord)
        # mask_logits: [1, num_queries_all, H/4, W/4]
        mask_f = []
        inst_st = 0
        for num_inst in num_insts:
            # [1, selected_queries, 1, H/4, W/4]
            mask_f.append(mask_logits[:, inst_st:inst_st +
                                      num_inst, :, :].unsqueeze(2))
            inst_st += num_inst

        output_pred_masks = torch.cat(mask_f, dim=0)

        return output_pred_masks

    def dynamic_mask_with_coords(self,
                                 mask_feats,
                                 reference_points,
                                 mask_head_params,
                                 num_insts,
                                 mask_feat_stride,
                                 rel_coord=True):

        device = mask_feats.device

        bs, in_channels, H, W = mask_feats.size()
        num_insts_all = reference_points.shape[1]

        locations = compute_locations(
            mask_feats.size(2),
            mask_feats.size(3),
            device=device,
            stride=mask_feat_stride)
        # locations: [H*W, 2]

        if rel_coord:
            instance_locations = reference_points
            relative_coords = instance_locations.reshape(
                1, num_insts_all, 1, 1, 2) - locations.reshape(1, 1, H, W, 2)
            relative_coords = relative_coords.float()
            relative_coords = relative_coords.permute(0, 1, 4, 2,
                                                      3).flatten(-2, -1)
            mask_head_inputs = []
            inst_st = 0
            for i, num_inst in enumerate(num_insts):
                # [1, num_queries * (C/32+2), H/8 * W/8]
                relative_coords_b = relative_coords[:, inst_st:inst_st +
                                                    num_inst, :, :]
                mask_feats_b = mask_feats[i].reshape(
                    1, in_channels,
                    H * W).unsqueeze(1).repeat(1, num_inst, 1, 1)
                mask_head_b = torch.cat([relative_coords_b, mask_feats_b],
                                        dim=2)

                mask_head_inputs.append(mask_head_b)
                inst_st += num_inst

        else:
            mask_head_inputs = []
            inst_st = 0
            for i, num_inst in enumerate(num_insts):
                mask_head_b = mask_feats[i].reshape(1, in_channels,
                                                    H * W).unsqueeze(1).repeat(
                                                        1, num_inst, 1, 1)
                mask_head_b = mask_head_b.reshape(1, -1, H, W)
                mask_head_inputs.append(mask_head_b)

        # mask_head_inputs: [1, \sum{num_queries * (C/32+2)}, H/8, W/8]
        mask_head_inputs = torch.cat(mask_head_inputs, dim=1)
        mask_head_inputs = mask_head_inputs.reshape(1, -1, H, W)

        # mask_head_params: [num_insts_all, num_params]
        mask_head_params = torch.flatten(mask_head_params, 0, 1)

        if num_insts_all != 0:
            weights, biases = parse_dynamic_params(mask_head_params,
                                                   self.dynamic_mask_channels,
                                                   self.weight_nums,
                                                   self.bias_nums)

            mask_logits = self.dynamic_conv_forward(mask_head_inputs, weights,
                                                    biases,
                                                    mask_head_params.shape[0])
        else:
            mask_logits = mask_head_inputs
            return mask_logits
        # mask_logits: [1, num_insts_all, H/8, W/8]
        mask_logits = mask_logits.reshape(-1, 1, H, W)

        # upsample predicted masks
        assert mask_feat_stride >= self.mask_out_stride
        assert mask_feat_stride % self.mask_out_stride == 0

        mask_logits = aligned_bilinear(
            mask_logits, int(mask_feat_stride / self.mask_out_stride))

        mask_logits = mask_logits.reshape(1, -1, mask_logits.shape[-2],
                                          mask_logits.shape[-1])
        # mask_logits: [1, num_insts_all, H/4, W/4]

        return mask_logits

    def dynamic_conv_forward(self, features: Tensor, weights: List[Tensor],
                             biases: List[Tensor], num_insts: int) -> Tensor:
        """dynamic forward, each layer follow a relu."""
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(x, w, bias=b, stride=1, padding=0, groups=num_insts)
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)

        outs = self(x, batch_img_metas)
        loss_inputs = outs + (batch_gt_instances, batch_img_metas)
        losses = self.loss_by_feat(*loss_inputs)
        return losses

    def loss_by_feat(
        self,
        all_cls_scores: Tensor,
        all_bbox_preds: Tensor,
        all_masks_preds: Tensor,
        embeds_preds: InstanceData,
        enc_cls_scores: Tensor,
        enc_bbox_preds: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:

        assert batch_gt_instances_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'
        # fix batch_img_metas
        key_list = [
            'batch_input_shape', 'pad_shape', 'img_shape', 'scale_factor'
        ]
        for key in key_list:
            for batch_id in range(len(batch_img_metas)):
                batch_img_metas[batch_id][key] = batch_img_metas[batch_id][
                    key][0]

        num_dec_layers = len(all_cls_scores)
        batch_key_gt_instances = []
        batch_ref_gt_instances = []
        for gt_instances in batch_gt_instances:
            batch_key_gt_instances.append(
                gt_instances[gt_instances.map_instances_to_img_idx == 0])
            batch_ref_gt_instances.append(
                gt_instances[gt_instances.map_instances_to_img_idx == 1])

        batch_key_gt_instances_list = [
            batch_key_gt_instances for _ in range(num_dec_layers)
        ]
        batch_ref_gt_instances_list = [
            batch_ref_gt_instances for _ in range(num_dec_layers)
        ]
        batch_img_metas_list = [batch_img_metas for _ in range(num_dec_layers)]

        losses_cls, losses_bbox, losses_iou, \
            losses_mask, losses_dice, min_cost_qids = multi_apply(
                self.loss_by_feat_single,
                all_cls_scores,
                all_bbox_preds,
                all_masks_preds,
                batch_key_gt_instances_list,
                batch_ref_gt_instances_list,
                batch_img_metas_list)

        # get track loss
        contrast_items = get_contrast_items(
            key_embeds=embeds_preds.key_embeds,
            ref_embeds=embeds_preds.ref_embeds,
            key_gt_instances=batch_key_gt_instances,
            ref_gt_instances=batch_ref_gt_instances,
            ref_bbox=embeds_preds.ref_bbox,  # rescale bboxes
            ref_cls=embeds_preds.ref_cls.sigmoid(),
            query_inds=min_cost_qids[-1],
            batch_img_metas=batch_img_metas)

        loss_track = 0.
        loss_track_aux = 0.
        for contrast_item in contrast_items:
            loss_track += self.loss_track(contrast_item['contrast'],
                                          contrast_item['label'])
            if self.loss_track_aux is not None:
                loss_track_aux += self.loss_track_aux(
                    contrast_item['aux_consin'], contrast_item['aux_label'])
        loss_track = loss_track / len(contrast_items)
        if self.loss_track_aux is not None:
            loss_track_aux = loss_track_aux / len(contrast_items)

        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            for i in range(len(batch_img_metas)):
                batch_gt_instances[i].labels = torch.zeros_like(
                    batch_gt_instances[i].labels)
            enc_loss_cls, enc_losses_bbox, enc_losses_iou = \
                self.loss_single(enc_cls_scores, enc_bbox_preds,
                                 batch_gt_instances, batch_img_metas)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox
            loss_dict['enc_loss_iou'] = enc_losses_iou

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_iou'] = losses_iou[-1]
        loss_dict['loss_mask'] = losses_mask[-1]
        loss_dict['loss_dice'] = losses_dice[-1]
        loss_dict['loss_track'] = loss_track
        loss_dict['loss_track_aux'] = loss_track_aux
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i, loss_mask_i, loss_dice_i in \
            zip(
                losses_cls[:-1], losses_bbox[:-1], losses_iou[:-1],
                losses_mask[:-1], losses_dice[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
            loss_dict[f'd{num_dec_layer}.loss_mask'] = loss_mask_i
            loss_dict[f'd{num_dec_layer}.loss_dice'] = loss_dice_i
            num_dec_layer += 1
        return loss_dict

    def loss_by_feat_single(self, cls_scores: Tensor, bbox_preds: Tensor,
                            mask_preds: Tensor,
                            batch_key_gt_instances: InstanceList,
                            batch_ref_gt_instances: InstanceList,
                            batch_img_metas: List[dict]) -> Tuple[Tensor]:

        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]

        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           mask_preds_list,
                                           batch_key_gt_instances,
                                           batch_img_metas)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         mask_targets_list, mask_weights_list, num_total_pos, num_total_neg,
         min_cost_qid) = cls_reg_targets

        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        mask_targets = torch.cat(mask_targets_list, 0)
        mask_weights = torch.cat(mask_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, bbox_preds):
            img_h, img_w, = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)

        # extract positive ones
        # shape (batch_size, num_queries, h, w) -> (num_total_gts, h, w)
        mask_preds = mask_preds.flatten(0, 1).squeeze(1)
        mask_preds = mask_preds[mask_weights > 0]

        if mask_targets.shape[0] == 0:
            # zero match
            loss_dice = mask_preds.sum()
            loss_mask = mask_preds.sum()
        else:
            # dice loss
            loss_dice = self.loss_dice(
                mask_preds, mask_targets, avg_factor=num_total_pos)

            # mask loss
            # FocalLoss support input of shape (n, num_class)
            h, w = mask_preds.shape[-2:]
            # shape (num_total_gts, h, w) -> (num_total_gts * h * w, 1)
            mask_preds = mask_preds.reshape(-1, 1)
            # shape (num_total_gts, h, w) -> (num_total_gts * h * w)
            mask_targets = mask_targets.reshape(-1)
            loss_mask = self.loss_mask(
                mask_preds, 1 - mask_targets, avg_factor=num_total_pos * h * w)

        return loss_cls, loss_bbox, loss_iou, \
            loss_mask, loss_dice, min_cost_qid

    def get_targets(self, cls_scores_list: List[Tensor],
                    bbox_preds_list: List[Tensor],
                    mask_preds_list: List[Tensor],
                    batch_gt_instances: InstanceList,
                    batch_img_metas: List[dict]) -> tuple:
        """Compute regression and classification targets for a batch image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            tuple: a tuple containing the following targets.

            - labels_list (list[Tensor]): Labels for all images.
            - label_weights_list (list[Tensor]): Label weights for all images.
            - bbox_targets_list (list[Tensor]): BBox targets for all images.
            - bbox_weights_list (list[Tensor]): BBox weights for all images.
            - num_total_pos (int): Number of positive samples in all images.
            - num_total_neg (int): Number of negative samples in all images.
        """
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         mask_targets_list, mask_weights_list, pos_inds_list, neg_inds_list,
         min_cost_qid) = multi_apply(self._get_targets_single, cls_scores_list,
                                     bbox_preds_list, mask_preds_list,
                                     batch_gt_instances, batch_img_metas)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, mask_targets_list, mask_weights_list,
                num_total_pos, num_total_neg, min_cost_qid)

    def _get_targets_single(self, cls_score: Tensor, bbox_pred: Tensor,
                            mask_pred: Tensor, gt_instances: InstanceData,
                            img_meta: dict) -> tuple:
        """Compute regression and classification targets for one image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for one image.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

            - labels (Tensor): Labels of each image.
            - label_weights (Tensor]): Label weights of each image.
            - bbox_targets (Tensor): BBox targets of each image.
            - bbox_weights (Tensor): BBox weights of each image.
            - pos_inds (Tensor): Sampled positive indices for each image.
            - neg_inds (Tensor): Sampled negative indices for each image.
        """
        img_h, img_w = img_meta['img_shape']
        factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
        num_bboxes = bbox_pred.size(0)
        # convert bbox_pred from xywh, normalized to xyxy, unnormalized
        bbox_pred = bbox_cxcywh_to_xyxy(bbox_pred)
        bbox_pred = bbox_pred * factor

        pred_instances = InstanceData(scores=cls_score, bboxes=bbox_pred)
        # assigner and sampler
        assign_result = self.assigner.assign(
            pred_instances=pred_instances,
            gt_instances=gt_instances,
            img_meta=img_meta)

        gt_bboxes = gt_instances.bboxes
        target_shape = mask_pred.shape[-2:]
        gt_masks = gt_instances.masks.to_tensor(
            dtype=torch.bool, device=gt_bboxes.device)
        if gt_masks.shape[0] > 0:
            gt_masks_downsampled = F.interpolate(
                gt_masks.unsqueeze(1).float(), target_shape,
                mode='nearest').squeeze(1).long()
        else:
            gt_masks_downsampled = gt_masks

        gt_labels = gt_instances.labels
        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds.long(), :]
        pos_gt_masks = gt_masks_downsampled[pos_assigned_gt_inds.long(), :]
        min_cost_qid = assign_result.get_extra_property('query_inds')

        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # mask targets
        mask_targets = torch.zeros_like(mask_pred.squeeze(1), dtype=torch.long)
        mask_targets[pos_inds] = pos_gt_masks
        mask_weights = gt_bboxes.new_zeros(num_bboxes)
        mask_weights[pos_inds] = 1.0

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        pos_gt_bboxes_normalized = pos_gt_bboxes / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        bbox_targets[pos_inds] = pos_gt_bboxes_targets
        return (labels, label_weights, bbox_targets, bbox_weights,
                mask_targets, mask_weights, pos_inds, neg_inds, min_cost_qid)

    def predict(self,
                x: Tuple[Tensor],
                batch_data_samples: SampleList,
                rescale: bool = True) -> InstanceList:
        """Perform forward propagation of the detection head and predict
        detection results on the features of the upstream network. Over-write
        because img_metas are needed as inputs for bbox_head.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image
            after the post process.
        """
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        outs = self.inference_forward(x, batch_img_metas)

        predictions = self.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, rescale=rescale)
        return predictions

    def predict_by_feat(self,
                        all_cls_scores: Tensor,
                        all_bbox_preds: Tensor,
                        all_mask_preds: Tensor,
                        all_embeds_preds: Tensor,
                        enc_cls_scores: Tensor,
                        enc_bbox_preds: Tensor,
                        batch_img_metas: List[Dict],
                        rescale: bool = False) -> InstanceList:

        result_list = []
        for img_id in range(len(batch_img_metas)):
            cls_score = all_cls_scores[img_id]
            bbox_pred = all_bbox_preds[img_id]
            mask_pred = all_mask_preds[img_id]
            embed_pred = all_embeds_preds[img_id]
            img_meta = batch_img_metas[img_id]
            results = self._predict_by_feat_single(cls_score, bbox_pred,
                                                   mask_pred, embed_pred,
                                                   img_meta, rescale)
            result_list.append(results)
        return result_list

    def _predict_by_feat_single(self,
                                cls_score: Tensor,
                                bbox_pred: Tensor,
                                mask_pred: Tensor,
                                embed_pred: Tensor,
                                img_meta: dict,
                                rescale: bool = True) -> InstanceData:

        assert len(cls_score) == len(bbox_pred)
        img_shape = img_meta['img_shape']

        cls_score = cls_score.sigmoid()
        max_score, _ = torch.max(cls_score, 1)
        bbox_pred = bbox_cxcywh_to_xyxy(bbox_pred)
        indices = torch.nonzero(
            max_score > self.inference_select_thres, as_tuple=False).squeeze(1)
        if len(indices) == 0:
            topkv, indices_top1 = torch.topk(cls_score.max(1)[0], k=1)
            indices_top1 = indices_top1[torch.argmax(topkv)]
            indices = [indices_top1.tolist()]
        else:
            nms_scores, idxs = torch.max(cls_score[indices], 1)
            boxes_before_nms = bbox_pred[indices]
            keep_indices = ops.batched_nms(boxes_before_nms, nms_scores, idxs,
                                           0.9)
            indices = indices[keep_indices]

        scores = torch.max(cls_score[indices], 1)[0]
        det_bboxes = bbox_pred[indices]
        det_labels = torch.argmax(cls_score[indices], dim=1)
        track_feats = embed_pred[indices]
        det_masks = mask_pred[indices]

        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])

        results = InstanceData()
        results.bboxes = det_bboxes
        results.scores = scores
        results.labels = det_labels
        results.masks = det_masks
        results.track_feats = track_feats
        return results
