# Copyright (c) OpenMMLab. All rights reserved.
import random
from typing import List

import torch
import torch.nn as nn
import torchvision.ops as ops
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh

from .utils import generalized_box_iou


def get_contrast_items(key_embeds, ref_embeds, key_gt_instances,
                       ref_gt_instances, ref_bbox, ref_cls, query_inds,
                       batch_img_metas) -> List:

    one = torch.tensor(1).to(ref_embeds)
    zero = torch.tensor(0).to(ref_embeds)
    contrast_items = []

    for bid, (key_gt, ref_gt, indices) in enumerate(
            zip(key_gt_instances, ref_gt_instances, query_inds)):

        key_ins_ids = key_gt.instances_id
        ref_ins_ids = ref_gt.instances_id
        valid = torch.tensor([(ref_ins_id in key_ins_ids)
                              for ref_ins_id in ref_ins_ids],
                             dtype=torch.bool)

        img_h, img_w, = batch_img_metas[bid]['img_shape']
        gt_bboxes = bbox_xyxy_to_cxcywh(ref_gt['bboxes'])
        factor = gt_bboxes.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0).repeat(
                                           gt_bboxes.size(0), 1)
        gt_bboxes /= factor
        gt_labels = ref_gt['labels']
        contrastive_pos, contrastive_neg = get_pos_idx(ref_bbox[bid],
                                                       ref_cls[bid], gt_bboxes,
                                                       gt_labels, valid)

        for inst_i, (is_valid, matched_query_id) in enumerate(
                zip(valid, query_inds[bid])):

            if not is_valid:
                continue
            # key_embeds: (bs, num_queries, c)
            key_embed_i = key_embeds[bid, matched_query_id].unsqueeze(0)

            pos_embed = ref_embeds[bid][contrastive_pos[inst_i]]
            neg_embed = ref_embeds[bid][~contrastive_neg[inst_i]]
            contrastive_embed = torch.cat([pos_embed, neg_embed], dim=0)
            contrastive_label = torch.cat(
                [one.repeat(len(pos_embed)),
                 zero.repeat(len(neg_embed))],
                dim=0).unsqueeze(1)

            contrast = torch.einsum('nc,kc->nk',
                                    [contrastive_embed, key_embed_i])

            if len(pos_embed) == 0:
                num_sample_neg = 10
            elif len(pos_embed) * 10 >= len(neg_embed):
                num_sample_neg = len(neg_embed)
            else:
                num_sample_neg = len(pos_embed) * 10

            # for aux loss
            sample_ids = random.sample(
                list(range(0, len(neg_embed))), num_sample_neg)
            aux_contrastive_embed = torch.cat(
                [pos_embed, neg_embed[sample_ids]], dim=0)
            aux_contrastive_label = torch.cat(
                [one.repeat(len(pos_embed)),
                 zero.repeat(num_sample_neg)],
                dim=0).unsqueeze(1)
            aux_contrastive_embed = nn.functional.normalize(
                aux_contrastive_embed.float(), dim=1)
            key_embed_i = nn.functional.normalize(key_embed_i.float(), dim=1)
            cosine = torch.einsum('nc,kc->nk',
                                  [aux_contrastive_embed, key_embed_i])

            contrast_items.append({
                'contrast': contrast,
                'label': contrastive_label,
                'aux_consin': cosine,
                'aux_label': aux_contrastive_label
            })

    return contrast_items


def get_pos_idx(ref_bbox, ref_cls, gt_bbox, gt_cls, valid):

    with torch.no_grad():
        if False in valid:
            gt_bbox = gt_bbox[valid]
            gt_cls = gt_cls[valid]

        fg_mask, is_in_boxes_and_center = \
            get_in_gt_and_in_center_info(
                ref_bbox,
                gt_bbox,
                expanded_strides=32)
        pair_wise_ious = ops.box_iou(
            bbox_cxcywh_to_xyxy(ref_bbox), bbox_cxcywh_to_xyxy(gt_bbox))

        # Compute the classification cost.
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (ref_cls ** gamma) * \
            (-(1 - ref_cls + 1e-8).log())
        pos_cost_class = alpha * (
            (1 - ref_cls)**gamma) * (-(ref_cls + 1e-8).log())
        cost_class = pos_cost_class[:, gt_cls] - neg_cost_class[:, gt_cls]
        cost_giou = - \
            generalized_box_iou(bbox_cxcywh_to_xyxy(
                ref_bbox),  bbox_cxcywh_to_xyxy(gt_bbox))

        cost = (
            cost_class + 3.0 * cost_giou + 100.0 * (~is_in_boxes_and_center))

        cost[~fg_mask] = cost[~fg_mask] + 10000.0

        if False in valid:
            pos_indices = []
            neg_indices = []
            if valid.sum() > 0:
                # Select positive sample on reference frame, k = 10
                pos_matched = dynamic_k_matching(cost, pair_wise_ious,
                                                 int(valid.sum()), 10)
                # Select positive sample on reference frame, k = 100
                neg_matched = dynamic_k_matching(cost, pair_wise_ious,
                                                 int(valid.sum()), 100)
            valid_idx = 0
            valid_list = valid.tolist()
            for istrue in valid_list:
                if istrue:
                    pos_indices.append(pos_matched[valid_idx])
                    neg_indices.append(neg_matched[valid_idx])
                    valid_idx += 1
                else:
                    pos_indices.append(None)
                    neg_indices.append(None)

        else:
            if valid.sum() > 0:
                pos_indices = dynamic_k_matching(cost, pair_wise_ious,
                                                 gt_bbox.shape[0], 10)
                neg_indices = dynamic_k_matching(cost, pair_wise_ious,
                                                 gt_bbox.shape[0], 100)
            else:
                pos_indices = [None]
                neg_indices = [None]

    return (pos_indices, neg_indices)


def get_in_gt_and_in_center_info(bboxes, target_gts, expanded_strides):

    xy_target_gts = bbox_cxcywh_to_xyxy(target_gts)

    anchor_center_x = bboxes[:, 0].unsqueeze(1)
    anchor_center_y = bboxes[:, 1].unsqueeze(1)

    b_l = anchor_center_x > xy_target_gts[:, 0].unsqueeze(0)
    b_r = anchor_center_x < xy_target_gts[:, 2].unsqueeze(0)
    b_t = anchor_center_y > xy_target_gts[:, 1].unsqueeze(0)
    b_b = anchor_center_y < xy_target_gts[:, 3].unsqueeze(0)
    is_in_boxes = ((b_l.long() + b_r.long() + b_t.long() + b_b.long()) == 4)
    is_in_boxes_all = is_in_boxes.sum(1) > 0

    # in fixed center
    center_radius = 2.5
    b_l = anchor_center_x > (
        target_gts[:, 0] - (1 * center_radius / expanded_strides)).unsqueeze(0)
    b_r = anchor_center_x < (
        target_gts[:, 0] + (1 * center_radius / expanded_strides)).unsqueeze(0)
    b_t = anchor_center_y > (
        target_gts[:, 1] - (1 * center_radius / expanded_strides)).unsqueeze(0)
    b_b = anchor_center_y < (
        target_gts[:, 1] + (1 * center_radius / expanded_strides)).unsqueeze(0)
    is_in_centers = ((b_l.long() + b_r.long() + b_t.long() + b_b.long()) == 4)
    is_in_centers_all = is_in_centers.sum(1) > 0

    # in boxes or in centers, shape: [num_priors]
    is_in_boxes_or_centers = is_in_boxes_all | is_in_centers_all
    # both in boxes and centers, shape: [num_fg, num_gt]
    is_in_boxes_and_center = (is_in_boxes & is_in_centers)

    return is_in_boxes_or_centers, is_in_boxes_and_center


def dynamic_k_matching(cost, pair_wise_ious, num_gt, n_candidate_k):
    matching_matrix = torch.zeros_like(cost)
    ious_in_boxes_matrix = pair_wise_ious

    topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=0)
    dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
    for gt_idx in range(num_gt):
        _, pos_idx = torch.topk(
            cost[:, gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
        matching_matrix[:, gt_idx][pos_idx] = 1.0

    del topk_ious, dynamic_ks, pos_idx

    anchor_matching_gt = matching_matrix.sum(1)

    if (anchor_matching_gt > 1).sum() > 0:
        _, cost_argmin = torch.min(cost[anchor_matching_gt > 1], dim=1)
        matching_matrix[anchor_matching_gt > 1] *= 0
        matching_matrix[anchor_matching_gt > 1, cost_argmin, ] = 1

    while (matching_matrix.sum(0) == 0).any():
        matched_query_id = matching_matrix.sum(1) > 0
        cost[matched_query_id] += 100000.0
        unmatch_id = torch.nonzero(
            matching_matrix.sum(0) == 0, as_tuple=False).squeeze(1)
        for gt_idx in unmatch_id:
            pos_idx = torch.argmin(cost[:, gt_idx])
            matching_matrix[:, gt_idx][pos_idx] = 1.0
        if (matching_matrix.sum(1) > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[anchor_matching_gt > 1], dim=1)
            matching_matrix[anchor_matching_gt > 1] *= 0
            matching_matrix[anchor_matching_gt > 1, cost_argmin, ] = 1

    assert not (matching_matrix.sum(0) == 0).any()

    matched_pos = []
    for gt_idx in range(num_gt):
        matched_pos.append(matching_matrix[:, gt_idx] > 0)

    return matched_pos
