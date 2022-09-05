# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional

import numpy as np
from mmdet.evaluation.functional import bbox_overlaps


def success_overlap(gt_bboxes: np.ndarray, pred_bboxes: np.ndarray,
                    iou_th: np.ndarray, video_length: int) -> np.ndarray:
    """Evaluation based on iou.

    Args:
        gt_bboxes (np.ndarray): of shape (video_length, 4) in
            [tl_x, tl_y, br_x, br_y] format.
        pred_bboxes (np.ndarray): of shape (video_length, 4) in
            [tl_x, tl_y, br_x, br_y] format.
        iou_th (np.ndarray): Different threshold of iou. Typically is set to
            `np.arange(0, 1.05, 0.05)`.
        video_length (int): Video length.

    Returns:
        np.ndarray: The evaluation results at different threshold of iou.
    """
    success = np.zeros(len(iou_th))
    iou = np.ones(len(gt_bboxes)) * (-1)
    valid = (gt_bboxes[:, 2] > gt_bboxes[:, 0]) & (
        gt_bboxes[:, 3] > gt_bboxes[:, 1])
    iou_matrix = bbox_overlaps(gt_bboxes[valid], pred_bboxes[valid])
    iou[valid] = iou_matrix[np.arange(len(gt_bboxes[valid])),
                            np.arange(len(gt_bboxes[valid]))]

    for i in range(len(iou_th)):
        success[i] = np.sum(iou > iou_th[i]) / float(video_length)
    return success


def success_error(gt_bboxes_center: np.ndarray, pred_bboxes_center: np.ndarray,
                  pixel_offset_th: np.ndarray,
                  video_length: int) -> np.ndarray:
    """Evaluation based on pixel offset.

    Args:
        gt_bboxes (np.ndarray): of shape (video_length, 2) in [cx, cy] format.
        pred_bboxes (np.ndarray): of shape (video_length, 2) in [cx, cy]
            format.
        pixel_offset_th (np.ndarray): Different threshold of pixel offset.
        video_length (int): Video length.

    Returns:
        np.ndarray: The evaluation results at different threshold of pixel
            offset.
    """
    success = np.zeros(len(pixel_offset_th))
    dist = np.ones(len(gt_bboxes_center)) * (-1)
    valid = (gt_bboxes_center[:, 0] > 0) & (gt_bboxes_center[:, 1] > 0)
    dist[valid] = np.sqrt(
        np.sum(
            (gt_bboxes_center[valid] - pred_bboxes_center[valid])**2, axis=1))
    for i in range(len(pixel_offset_th)):
        success[i] = np.sum(dist <= pixel_offset_th[i]) / float(video_length)
    return success


def eval_sot_ope(
        results,
        annotations: List[List[np.ndarray]],
        visible_infos: Optional[List[np.ndarray]] = None) -> Dict[str, float]:
    """Evaluation in OPE protocol.

    Args:
        results (List[List[np.ndarray]]): The first list contains the tracking
            results of each video. The second list contains the tracking
            results of each frame in one video. The ndarray denotes the
            tracking box in [tl_x, tl_y, br_x, br_y] format.
        annotations (List[np.ndarray]): The list contains the bbox
            annotations of each video. The ndarray is gt_bboxes of one video.
            It's in (N, 4) shape. Each bbox is in (x1, y1, x2, y2) format.
        visible_infos (Optional[List[np.ndarray]], optional): If not None, the
            list contains the visible information of each video. The ndarray is
            visibility (with bool type) of object in one video. It's in (N,)
            shape. Default to None.

    Returns:
        Dict[str, float]: OPE style evaluation metric (i.e. success,
        norm precision and precision).
    """
    success_results = []
    precision_results = []
    norm_precision_results = []
    if visible_infos is None:
        visible_infos = [np.array([True] * len(_)) for _ in annotations]
    for single_video_results, single_video_gt_bboxes, single_video_visible in zip(  # noqa
            results, annotations, visible_infos):
        pred_bboxes = np.stack(single_video_results)
        assert len(pred_bboxes) == len(single_video_gt_bboxes)
        video_length = len(single_video_results)

        gt_bboxes = single_video_gt_bboxes[single_video_visible]
        pred_bboxes = pred_bboxes[single_video_visible]

        # eval success based on iou
        iou_th = np.arange(0, 1.05, 0.05)
        success_results.append(
            success_overlap(gt_bboxes, pred_bboxes, iou_th, video_length))

        # eval precision
        gt_bboxes_center = np.array(
            (0.5 * (gt_bboxes[:, 2] + gt_bboxes[:, 0]),
             0.5 * (gt_bboxes[:, 3] + gt_bboxes[:, 1]))).T
        pred_bboxes_center = np.array(
            (0.5 * (pred_bboxes[:, 2] + pred_bboxes[:, 0]),
             0.5 * (pred_bboxes[:, 3] + pred_bboxes[:, 1]))).T
        pixel_offset_th = np.arange(0, 51, 1)
        precision_results.append(
            success_error(gt_bboxes_center, pred_bboxes_center,
                          pixel_offset_th, video_length))

        # eval normed precision
        gt_bboxes_wh = np.array((gt_bboxes[:, 2] - gt_bboxes[:, 0],
                                 gt_bboxes[:, 3] - gt_bboxes[:, 1])).T
        norm_gt_bboxes_center = gt_bboxes_center / (gt_bboxes_wh + 1e-16)
        norm_pred_bboxes_center = pred_bboxes_center / (gt_bboxes_wh + 1e-16)
        norm_pixel_offset_th = pixel_offset_th / 100.
        norm_precision_results.append(
            success_error(norm_gt_bboxes_center, norm_pred_bboxes_center,
                          norm_pixel_offset_th, video_length))

    success_results = np.stack(success_results) * 100
    precision_results = np.stack(precision_results) * 100
    norm_precision_results = np.stack(norm_precision_results) * 100
    success = np.mean(success_results)
    precision = np.mean(precision_results, axis=0)[20]
    norm_precision = np.mean(norm_precision_results, axis=0)[20]
    eval_results = dict(
        success=success,
        norm_precision=norm_precision,
        precision=precision,
        ori_success=success_results,
        ori_precision=precision_results,
        ori_norm_precision=norm_precision_results)
    return eval_results
