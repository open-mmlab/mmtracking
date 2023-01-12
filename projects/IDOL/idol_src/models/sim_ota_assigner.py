# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch
import torchvision.ops as ops
from mmdet.models.task_modules import AssignResult
from mmdet.models.task_modules import SimOTAAssigner as MMDET_SimOTAAssigner
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy
from mmengine import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor

from mmtrack.registry import TASK_UTILS

INF = 10000.0
EPS = 1.0e-7


@TASK_UTILS.register_module()
class SimOTAAssigner(MMDET_SimOTAAssigner):
    """Computes matching between predictions and ground truth.

    Args:
        center_radius (float): Ground truth center size
            to judge whether a prior is in center. Defaults to 2.5.
        candidate_topk (int): The candidate top-k which used to
            get top-k ious to calculate dynamic-k. Defaults to 10.
        iou_weight (float): The scale factor for regression
            iou cost. Defaults to 3.0.
        cls_weight (float): The scale factor for classification
            cost. Defaults to 1.0.
        iou_calculator (ConfigType): Config of overlaps Calculator.
            Defaults to dict(type='BboxOverlaps2D').
    """

    def __init__(self,
                 match_costs: Union[List[Union[dict, ConfigDict]], dict,
                                    ConfigDict],
                 center_radius: float = 2.5,
                 candidate_topk: int = 10):

        if isinstance(match_costs, dict):
            match_costs = [match_costs]
        elif isinstance(match_costs, list):
            assert len(match_costs) > 0, \
                'match_costs must not be a empty list.'

        self.center_radius = center_radius
        self.candidate_topk = candidate_topk
        self.match_costs = [
            TASK_UTILS.build(match_cost) for match_cost in match_costs
        ]

    def assign(self,
               pred_instances: InstanceData,
               gt_instances: InstanceData,
               img_meta: Optional[dict] = None,
               **kwargs) -> AssignResult:

        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        num_gt = gt_bboxes.size(0)

        pred_bboxes = pred_instances.bboxes
        priors = pred_instances.bboxes
        num_bboxes = pred_bboxes.size(0)

        # assign 0 by default
        assigned_gt_inds = pred_bboxes.new_full((num_bboxes, ),
                                                0,
                                                dtype=torch.long)
        assigned_labels = pred_bboxes.new_full((num_bboxes, ),
                                               0,
                                               dtype=torch.long)
        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gt == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts=num_gt,
                gt_inds=assigned_gt_inds,
                max_overlaps=None,
                labels=assigned_labels)

        valid_mask, is_in_boxes_and_center = self.get_in_gt_and_in_center_info(
            priors, gt_bboxes)
        pairwise_ious = ops.box_iou(
            bbox_cxcywh_to_xyxy(pred_bboxes), bbox_cxcywh_to_xyxy(gt_bboxes))

        # compute weighted cost
        cost_list = []
        for match_cost in self.match_costs:
            cost = match_cost(
                pred_instances=pred_instances,
                gt_instances=gt_instances,
                img_meta=img_meta)
            cost_list.append(cost)
        cost = torch.stack(cost_list).sum(dim=0)
        cost += 100 * (~is_in_boxes_and_center)
        cost[~valid_mask] = cost[~valid_mask] + INF

        fg_mask_inboxes, matched_gt_inds, min_cost_qid = \
            self.dynamic_k_matching(
                cost, pairwise_ious, num_gt)

        # convert to AssignResult format
        assigned_gt_inds[fg_mask_inboxes] = matched_gt_inds + 1
        assigned_labels[fg_mask_inboxes] = gt_labels[matched_gt_inds].long()

        assign_res = AssignResult(
            num_gt, assigned_gt_inds, None, labels=assigned_labels)
        assign_res.set_extra_property('query_inds', min_cost_qid)
        return assign_res

    def get_in_gt_and_in_center_info(
            self, priors: Tensor, gt_bboxes: Tensor) -> Tuple[Tensor, Tensor]:
        """Get the information of which prior is in gt bboxes and gt center
        priors."""
        num_gt = gt_bboxes.size(0)

        repeated_x = priors[:, 0].unsqueeze(1).repeat(1, num_gt)
        repeated_y = priors[:, 1].unsqueeze(1).repeat(1, num_gt)
        repeated_stride_x = priors[:, 2].unsqueeze(1).repeat(1, num_gt)
        repeated_stride_y = priors[:, 3].unsqueeze(1).repeat(1, num_gt)

        # is prior centers in gt bboxes, shape: [n_prior, n_gt]
        l_ = repeated_x - gt_bboxes[:, 0]
        t_ = repeated_y - gt_bboxes[:, 1]
        r_ = gt_bboxes[:, 2] - repeated_x
        b_ = gt_bboxes[:, 3] - repeated_y

        deltas = torch.stack([l_, t_, r_, b_], dim=1)
        is_in_gts = deltas.min(dim=1).values > 0
        is_in_gts_all = is_in_gts.sum(dim=1) > 0

        # is prior centers in gt centers
        gt_cxs = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
        gt_cys = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
        ct_box_l = gt_cxs - self.center_radius * repeated_stride_x
        ct_box_t = gt_cys - self.center_radius * repeated_stride_y
        ct_box_r = gt_cxs + self.center_radius * repeated_stride_x
        ct_box_b = gt_cys + self.center_radius * repeated_stride_y

        cl_ = repeated_x - ct_box_l
        ct_ = repeated_y - ct_box_t
        cr_ = ct_box_r - repeated_x
        cb_ = ct_box_b - repeated_y

        ct_deltas = torch.stack([cl_, ct_, cr_, cb_], dim=1)
        is_in_cts = ct_deltas.min(dim=1).values > 0
        is_in_cts_all = is_in_cts.sum(dim=1) > 0

        # in boxes or in centers, shape: [num_priors]
        is_in_gts_or_centers = is_in_gts_all | is_in_cts_all

        # both in boxes and centers, shape: [num_fg, num_gt]
        is_in_boxes_and_centers = (is_in_gts & is_in_cts)
        return is_in_gts_or_centers, is_in_boxes_and_centers

    def dynamic_k_matching(self, cost: Tensor, pairwise_ious: Tensor,
                           num_gt: int) -> Tuple[Tensor, Tensor]:
        """Use IoU and matching cost to calculate the dynamic top-k positive
        targets."""
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)
        # select candidate topk ious for dynamic-k calculation
        candidate_topk = min(self.candidate_topk, pairwise_ious.size(0))
        topk_ious, _ = torch.topk(pairwise_ious, candidate_topk, dim=0)
        # calculate dynamic k for each gt
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[:, gt_idx], k=dynamic_ks[gt_idx], largest=False)
            matching_matrix[:, gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        prior_match_gt_mask = matching_matrix.sum(1) > 1
        if prior_match_gt_mask.sum() > 0:
            cost_min, cost_argmin = torch.min(
                cost[prior_match_gt_mask, :], dim=1)
            matching_matrix[prior_match_gt_mask, :] *= 0
            matching_matrix[prior_match_gt_mask, cost_argmin] = 1

        while (matching_matrix.sum(0) == 0).any():
            matched_query_id = matching_matrix.sum(1) > 0
            cost[matched_query_id] += 100000.0
            unmatch_id = torch.nonzero(
                matching_matrix.sum(0) == 0, as_tuple=False).squeeze(1)
            for gt_idx in unmatch_id:
                pos_idx = torch.argmin(cost[:, gt_idx])
                matching_matrix[:, gt_idx][pos_idx] = 1.0
            if (matching_matrix.sum(1) >
                    1).sum() > 0:  # If a query matches more than one gt
                # find gt for these queries with minimal cost
                _, cost_argmin = torch.min(
                    cost[prior_match_gt_mask > 1], dim=1)
                # reset mapping relationship
                matching_matrix[prior_match_gt_mask > 1] *= 0
                # keep gt with minimal cost
                matching_matrix[prior_match_gt_mask > 1, cost_argmin, ] = 1

        assert not (matching_matrix.sum(0) == 0).any()
        # get foreground mask inside box and center prior
        fg_mask_inboxes = matching_matrix.sum(1) > 0
        matched_gt_inds = matching_matrix[fg_mask_inboxes, :].argmax(1)

        cost[matching_matrix == 0] = cost[matching_matrix == 0] + float('inf')
        min_cost_query_id = torch.min(cost, dim=0)[1]

        return fg_mask_inboxes, matched_gt_inds, min_cost_query_id
