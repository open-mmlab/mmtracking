# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import torch
from mmdet.core import bbox2result


def imrenormalize(img, img_norm_cfg, new_img_norm_cfg):
    """Re-normalize the image.

    Args:
        img (Tensor | ndarray): Input image. If the input is a Tensor, the
            shape is (1, C, H, W). If the input is a ndarray, the shape
            is (H, W, C).
        img_norm_cfg (dict): Original configuration for the normalization.
        new_img_norm_cfg (dict): New configuration for the normalization.

    Returns:
        Tensor | ndarray: Output image with the same type and shape of
        the input.
    """
    if isinstance(img, torch.Tensor):
        assert img.ndim == 4 and img.shape[0] == 1
        new_img = img.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        new_img = _imrenormalize(new_img, img_norm_cfg, new_img_norm_cfg)
        new_img = new_img.transpose(2, 0, 1)[None]
        return torch.from_numpy(new_img).to(img)
    else:
        return _imrenormalize(img, img_norm_cfg, new_img_norm_cfg)


def _imrenormalize(img, img_norm_cfg, new_img_norm_cfg):
    """Re-normalize the image."""
    img_norm_cfg = img_norm_cfg.copy()
    new_img_norm_cfg = new_img_norm_cfg.copy()
    for k, v in img_norm_cfg.items():
        if (k == 'mean' or k == 'std') and not isinstance(v, np.ndarray):
            img_norm_cfg[k] = np.array(v, dtype=img.dtype)
    # reverse cfg
    if 'to_rgb' in img_norm_cfg:
        img_norm_cfg['to_bgr'] = img_norm_cfg['to_rgb']
        img_norm_cfg.pop('to_rgb')
    for k, v in new_img_norm_cfg.items():
        if (k == 'mean' or k == 'std') and not isinstance(v, np.ndarray):
            new_img_norm_cfg[k] = np.array(v, dtype=img.dtype)
    img = mmcv.imdenormalize(img, **img_norm_cfg)
    img = mmcv.imnormalize(img, **new_img_norm_cfg)
    return img


def outs2results(output_dict):
    """Convert tracking/detection results to a list of numpy arrays.

    Args:
        output_dict (dict): The output results of the model forward. It may
            contain keys as belows:

            - bboxes (torch.Tensor | np.ndarray): shape (n, 5)
            - labels (torch.Tensor | np.ndarray): shape (n, )
            - masks (torch.Tensor | np.ndarray): shape (n, h, w)
            - ids (torch.Tensor | np.ndarray): shape (n, )
            - num_classes (int): class number, not including background class

    Returns:
        dict[str : list(ndarray) | list[list[np.ndarray]]]: tracking/detection
        results of each class. It may contain keys as belows:

        - bboxes (list[np.ndarray]): Each list denotes bboxes of one
            category.
        - masks (list[list[np.ndarray]]): Each outer list denotes masks of
            one category. Each inner list denotes one mask belonging to
            the category. Each mask has shape (h, w).
    """
    for key in output_dict:
        assert key in ['bboxes', 'labels', 'masks', 'ids', 'num_classes']
    for key in ['labels', 'num_classes']:
        assert key in output_dict

    bboxes = output_dict.get('bboxes', None)
    masks = output_dict.get('masks', None)
    ids = output_dict.get('ids', None)
    labels = output_dict['labels']
    num_classes = output_dict['num_classes']
    result_dict = dict()

    if ids is not None:
        valid_inds = ids > -1
        ids = ids[valid_inds]
        labels = labels[valid_inds]

    if bboxes is not None:
        if ids is not None:
            bboxes = bboxes[valid_inds]
            if bboxes.shape[0] == 0:
                bbox_results = [
                    np.zeros((0, 6), dtype=np.float32)
                    for i in range(num_classes)
                ]
            else:
                if isinstance(bboxes, torch.Tensor):
                    bboxes = bboxes.cpu().numpy()
                    labels = labels.cpu().numpy()
                    ids = ids.cpu().numpy()
                bbox_results = [
                    np.concatenate(
                        (ids[labels == i, None], bboxes[labels == i, :]),
                        axis=1) for i in range(num_classes)
                ]
        else:
            bbox_results = bbox2result(bboxes, labels, num_classes)
        result_dict['bboxes'] = bbox_results

    if masks is not None:
        if ids is not None:
            masks = masks[valid_inds]
        if isinstance(masks, torch.Tensor):
            masks = masks.detach().cpu().numpy()
        masks_results = [[] for _ in range(num_classes)]
        for i in range(bboxes.shape[0]):
            masks_results[labels[i]].append(masks[i])
        result_dict['masks'] = masks_results

    return result_dict


def results2outs(result_dict):
    """Restore the results (list of results of each category) into the results
    of the model forward.

    Args:
        result_dict (dict): List of results of each category. It may
            contain keys as belows:

            - bboxes (list[np.ndarray]): Each list denotes bboxes of one
                category.
            - masks (list[list[np.ndarray]]): Each outer list denotes masks of
                one category. Each inner list denotes one mask belonging to
                the category. Each mask has shape (h, w).
            - mask_shape (tuple[int]): The shape (h, w) of mask.

    Returns:
        tuple: tracking results of each class. It may contain keys as belows:

        - bboxes (np.ndarray): shape (n, 5)
        - labels (np.ndarray): shape (n, )
        - masks (np.ndarray): shape (n, h, w)
        - ids (np.ndarray): shape (n, )
    """
    for key in result_dict:
        assert key in ['bboxes', 'masks', 'mask_shape']
    bbox_results = result_dict.get('bboxes', None)
    mask_results = result_dict.get('masks', None)
    track_dict = dict()

    if bbox_results is not None:
        labels = []
        for i, bbox in enumerate(bbox_results):
            labels.extend([i] * bbox.shape[0])
        labels = np.array(labels, dtype=np.int64)
        track_dict['labels'] = labels

        bboxes = np.concatenate(bbox_results, axis=0).astype(np.float32)
        if bboxes.shape[1] == 5:
            track_dict['bboxes'] = bboxes
        elif bboxes.shape[1] == 6:
            ids = bboxes[:, 0].astype(np.int64)
            bboxes = bboxes[:, 1:]
            track_dict['bboxes'] = bboxes
            track_dict['ids'] = ids
        else:
            raise NotImplementedError(
                f'Not supported bbox shape: (N, {bboxes.shape[1]})')

    if mask_results is not None:
        assert 'mask_shape' in result_dict
        mask_height, mask_width = result_dict['mask_shape']
        mask_results = mmcv.concat_list(mask_results)
        if len(mask_results) == 0:
            masks = np.zeros((0, mask_height, mask_width)).astype(bool)
        else:
            masks = np.stack(mask_results, axis=0)
        track_dict['masks'] = masks

    return track_dict
