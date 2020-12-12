import mmcv
import numpy as np
import torch


def imrenormalize(img, img_norm_cfg, new_img_norm_cfg):
    if isinstance(img, torch.Tensor):
        assert img.ndim == 4 and img.shape[0] == 1
        new_img = img.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        new_img = _imrenormalize(new_img, img_norm_cfg, new_img_norm_cfg)
        new_img = new_img.transpose(2, 0, 1)[None]
        return torch.from_numpy(new_img).to(img.device)
    else:
        return _imrenormalize(img, img_norm_cfg, new_img_norm_cfg)


def _imrenormalize(img, img_norm_cfg, new_img_norm_cfg):
    for k, v in img_norm_cfg.items():
        if (k == 'mean' or k == 'std') and not isinstance(v, np.ndarray):
            img_norm_cfg[k] = np.array(v, dtype=np.float32)
    # reverse cfg
    if 'to_rgb' in img_norm_cfg:
        img_norm_cfg['to_bgr'] = img_norm_cfg['to_rgb']
        img_norm_cfg.pop('to_rgb')
    for k, v in new_img_norm_cfg.items():
        if (k == 'mean' or k == 'std') and not isinstance(v, np.ndarray):
            new_img_norm_cfg[k] = np.array(v, dtype=np.float32)
    img = mmcv.imdenormalize(img, **img_norm_cfg)
    img = mmcv.imnormalize(img, **new_img_norm_cfg)
    return img


def track2result(bboxes, labels, ids, num_classes):
    valid_inds = ids > -1
    bboxes = bboxes[valid_inds]
    labels = labels[valid_inds]
    ids = ids[valid_inds]

    if bboxes.shape[0] == 0:
        return [np.zeros(bboxes.shape) for i in range(num_classes)]
    else:
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.cpu().numpy()
            labels = labels.cpu().numpy()
            ids = ids.cpu().numpy()
        return [
            np.concatenate((ids[labels == i, None], bboxes[labels == i, :]),
                           axis=1) for i in range(num_classes)
        ]


def restore_result(result, return_ids=False):
    labels = []
    for i, bbox in enumerate(result):
        labels.extend([i] * bbox.shape[0])
    bboxes = np.concatenate(result, axis=0).astype(np.float32)
    labels = np.array(labels, dtype=np.int64)
    if return_ids:
        ids = bboxes[:, 0].astype(np.int64)
        bboxes = bboxes[:, 1:]
        return bboxes, labels, ids
    else:
        return bboxes, labels
