# Copyright (c) OpenMMLab. All rights reserved.
import copy
from os.path import dirname, exists, join

import numpy as np
import torch
from mmdet.utils.util_random import ensure_rng
from mmengine.dataset import pseudo_collate
from mmengine.structures import InstanceData

from mmtrack.structures import TrackDataSample


def random_boxes(num=1, scale=1, rng=None):
    """Simple version of ``kwimage.Boxes.random``

    Returns:
        Tensor: shape (n, 4) in x1, y1, x2, y2 format.

    References:
        https://gitlab.kitware.com/computer-vision/kwimage/blob/master/kwimage/structs/boxes.py#L1390 # noqa: E501

    Example:
        >>> num = 3
        >>> scale = 512
        >>> rng = 0
        >>> boxes = random_boxes(num, scale, rng)
        >>> print(boxes)
        tensor([[280.9925, 278.9802, 308.6148, 366.1769],
                [216.9113, 330.6978, 224.0446, 456.5878],
                [405.3632, 196.3221, 493.3953, 270.7942]])
    """
    rng = ensure_rng(rng)

    tlbr = rng.rand(num, 4).astype(np.float32)

    tl_x = np.minimum(tlbr[:, 0], tlbr[:, 2])
    tl_y = np.minimum(tlbr[:, 1], tlbr[:, 3])
    br_x = np.maximum(tlbr[:, 0], tlbr[:, 2])
    br_y = np.maximum(tlbr[:, 1], tlbr[:, 3])

    tlbr[:, 0] = tl_x * scale
    tlbr[:, 1] = tl_y * scale
    tlbr[:, 2] = br_x * scale
    tlbr[:, 3] = br_y * scale

    boxes = torch.from_numpy(tlbr)
    return boxes


def _get_config_directory():
    """Find the predefined video detector or tracker config directory."""
    try:
        # Assume we are running in the source mmtracking repo
        repo_dpath = dirname(dirname(dirname(__file__)))
    except NameError:
        # For IPython development when this __file__ is not defined
        import mmtrack
        repo_dpath = dirname(dirname(mmtrack.__file__))
    config_dpath = join(repo_dpath, 'configs')
    if not exists(config_dpath):
        raise Exception('Cannot find config path')
    return config_dpath


def _get_config_module(fname):
    """Load a configuration as a python module."""
    from mmengine import Config
    config_dpath = _get_config_directory()
    config_fpath = join(config_dpath, fname)
    config_mod = Config.fromfile(config_fpath)
    return config_mod


def get_model_cfg(fname):
    """Grab configs necessary to create a video detector or tracker.

    These are deep copied to allow for safe modification of parameters without
    influencing other tests.
    """
    config = _get_config_module(fname)
    model = copy.deepcopy(config.model)
    return model


def _rand_bboxes(rng, num_boxes, w, h):
    cx, cy, bw, bh = rng.rand(num_boxes, 4).T

    tl_x = ((cx * w) - (w * bw / 2)).clip(0, w)
    tl_y = ((cy * h) - (h * bh / 2)).clip(0, h)
    br_x = ((cx * w) + (w * bw / 2)).clip(0, w)
    br_y = ((cy * h) + (h * bh / 2)).clip(0, h)

    bboxes = np.vstack([tl_x, tl_y, br_x, br_y]).T
    return bboxes


def _rand_masks(rng, num_boxes, bboxes, img_w, img_h):
    from mmdet.structures.mask import BitmapMasks
    masks = np.zeros((num_boxes, img_h, img_w))
    for i, bbox in enumerate(bboxes):
        bbox = bbox.astype(np.int32)
        mask = (rng.rand(1, bbox[3] - bbox[1], bbox[2] - bbox[0]) >
                0.3).astype(np.int)
        masks[i:i + 1, bbox[1]:bbox[3], bbox[0]:bbox[2]] = mask
    return BitmapMasks(masks, height=img_h, width=img_w)


def demo_mm_inputs(batch_size=1,
                   frame_id=0,
                   num_key_imgs=1,
                   num_ref_imgs=1,
                   image_shapes=[(3, 128, 128)],
                   num_items=None,
                   num_classes=10,
                   ref_prefix='ref',
                   num_template_imgs=None,
                   num_search_imgs=None,
                   with_mask=False,
                   with_semantic=False):
    """Create a superset of inputs needed to run test or train batches.

    Args:
        batch_size (int): batch size. Default to 2.
        frame_id (int): the frame id.
        num_key_imgs (int): the number of key images. This input is used in
            all methods except for training in SOT.
        num_ref_imgs (int): the number of reference images. This input is
            used in all methods except for training in SOT.
        image_shapes (List[tuple], Optional): image shape.
            Default to (3, 128, 128)
        num_items (None | List[int]): specifies the number
            of boxes in each batch item. Default to None.
        num_classes (int): number of different labels a
            box might have. Default to 10.
        ref_prefix (str): the prefix of reference images (or search images
            in SOT).
        with_mask (bool): Whether to return mask annotation.
            Defaults to False.
        with_semantic (bool): whether to return semantic.
            Default to False.
        num_template_imgs (int): the number of template images. This input is
            only used in training in SOT.
        num_search_imgs (int): the number of search images. This input is
            only used in training in SOT.
    """
    # Compatible the names of one image group in SOT. `ref_prefix` means the
    # prefix of search images in SOT.
    assert (num_template_imgs is None) == (num_search_imgs is None)
    if num_template_imgs is not None:
        num_key_imgs, num_ref_imgs = num_template_imgs, num_search_imgs

    rng = np.random.RandomState(0)

    # Make sure the length of image_shapes is equal to ``batch_size``
    if isinstance(image_shapes, list):
        assert len(image_shapes) == batch_size
    else:
        image_shapes = [image_shapes] * batch_size

    # Make sure the length of each element in image_shapes is equal to
    # the number of types of image shapes since key_img and ref_image may have
    # different shapes.
    # After these transforms, as for ``image_shapes``, the
    # length of the outer list is equal to ``batch_size`` and the length of the
    # inner list is equal to the type of image shapes.
    num_img_group = int((num_key_imgs > 0) + (num_ref_imgs > 0))
    if isinstance(image_shapes[0], list):
        assert len(image_shapes[0]) == num_img_group and isinstance(
            image_shapes[0][0], tuple)
    else:
        assert isinstance(image_shapes[0], tuple)
        image_shapes = [[shape] * num_img_group for shape in image_shapes]

    if isinstance(num_items, list):
        assert len(num_items) == batch_size

    packed_inputs = []
    for idx in range(batch_size):
        image_shape_group = image_shapes[idx]
        c, h, w = image_shape_group[0]

        mm_inputs = dict(inputs=dict())
        if num_key_imgs > 0:
            key_img = rng.randint(
                0,
                255,
                size=(num_key_imgs, *image_shape_group[0]),
                dtype=np.uint8)
            mm_inputs['inputs']['img'] = torch.from_numpy(key_img)
        if num_ref_imgs > 0:
            index = int(num_key_imgs > 0)
            ref_img = rng.randint(
                0,
                255,
                size=(num_ref_imgs, *image_shape_group[index]),
                dtype=np.uint8)
            mm_inputs['inputs'][f'{ref_prefix}_img'] = torch.from_numpy(
                ref_img)

        img_meta = {
            'img_id': idx,
            'img_shape': image_shape_group[0][-2:],
            'ori_shape': image_shape_group[0][-2:],
            'filename': '<demo>.png',
            'scale_factor': np.array([1.1, 1.2]),
            'flip': False,
            'flip_direction': None,
            'is_video_data': True,
            'frame_id': frame_id
        }
        if num_ref_imgs > 0:
            search_img_meta = dict()
            for key, value in img_meta.items():
                search_img_meta[f'{ref_prefix}_{key}'] = [
                    value
                ] * num_ref_imgs if num_ref_imgs > 1 else value
            search_shape = image_shape_group[int(num_key_imgs > 0)][-2:]
            search_img_meta[f'{ref_prefix}_img_shape'] = [
                search_shape
            ] * num_ref_imgs if num_ref_imgs > 1 else search_shape
            search_img_meta[f'{ref_prefix}_ori_shape'] = [
                search_shape
            ] * num_ref_imgs if num_ref_imgs > 1 else search_shape
            img_meta.update(search_img_meta)

        data_sample = TrackDataSample()
        data_sample.set_metainfo(img_meta)

        # gt_instances
        gt_instances = InstanceData()
        if num_items is None:
            num_boxes = rng.randint(1, 10)
        else:
            num_boxes = num_items[idx]

        bboxes = _rand_bboxes(rng, num_boxes, w, h)
        labels = rng.randint(0, num_classes, size=num_boxes)
        instances_id = rng.randint(100, num_classes + 100, size=num_boxes)
        gt_instances.bboxes = torch.FloatTensor(bboxes)
        gt_instances.labels = torch.LongTensor(labels)
        gt_instances.instances_id = torch.LongTensor(instances_id)

        if with_mask:
            masks = _rand_masks(rng, num_boxes, bboxes, w, h)
            gt_instances.masks = masks

        data_sample.gt_instances = gt_instances
        # ignore_instances
        ignore_instances = InstanceData()
        bboxes = _rand_bboxes(rng, num_boxes, w, h)
        ignore_instances.bboxes = bboxes
        data_sample.ignored_instances = ignore_instances

        if num_ref_imgs > 0:
            ref_gt_instances = copy.deepcopy(gt_instances)
            setattr(data_sample, f'{ref_prefix}_gt_instances',
                    ref_gt_instances)
            ref_ignored_instances = copy.deepcopy(ignore_instances)
            setattr(data_sample, f'{ref_prefix}_ignored_instances',
                    ref_ignored_instances)

        mm_inputs['data_samples'] = data_sample

        # TODO: gt_ignore

        packed_inputs.append(mm_inputs)
    data = pseudo_collate(packed_inputs)
    return data
