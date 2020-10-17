import copy
from collections import defaultdict
from os.path import dirname, exists, join

import numpy as np
import pytest
import torch


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
    from mmcv import Config
    config_dpath = _get_config_directory()
    config_fpath = join(config_dpath, fname)
    config_mod = Config.fromfile(config_fpath)
    return config_mod


def _get_model_cfg(fname):
    """Grab configs necessary to create a video detector or tracker.

    These are deep copied to allow for safe modification of parameters without
    influencing other tests.
    """
    import mmcv
    config = _get_config_module(fname)
    model = copy.deepcopy(config.model)
    train_cfg = mmcv.Config(copy.deepcopy(config.train_cfg))
    test_cfg = mmcv.Config(copy.deepcopy(config.test_cfg))
    return model, train_cfg, test_cfg


@pytest.mark.parametrize('cfg_file',
                         ['vid/dff_faster_rcnn_r101_fpn_1x_imagenetvid.py'])
def test_vid_forward(cfg_file):
    model, train_cfg, test_cfg = _get_model_cfg(cfg_file)
    model['detector']['pretrained'] = None
    model['motion']['pretrained'] = None

    from mmtrack.models import build_model
    detector = build_model(model, train_cfg=train_cfg, test_cfg=test_cfg)

    input_shape = (1, 3, 256, 256)

    # Test forward train with a non-empty truth batch
    mm_inputs = _demo_mm_inputs(input_shape, num_items=[10])
    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')
    img_metas[0]['is_video_data'] = True
    gt_bboxes = mm_inputs['gt_bboxes']
    gt_labels = mm_inputs['gt_labels']
    gt_masks = mm_inputs['gt_masks']
    losses = detector.forward(
        img=imgs,
        img_metas=img_metas,
        gt_bboxes=gt_bboxes,
        gt_labels=gt_labels,
        ref_img=imgs.clone()[:, None, ...],
        ref_img_metas=copy.deepcopy(img_metas),
        ref_gt_bboxes=copy.deepcopy(gt_bboxes),
        ref_gt_labels=copy.deepcopy(gt_labels),
        gt_masks=gt_masks,
        ref_gt_masks=copy.deepcopy(gt_masks),
        return_loss=True)
    assert isinstance(losses, dict)
    loss, _ = detector._parse_losses(losses)
    loss.requires_grad_(True)
    assert float(loss.item()) > 0
    loss.backward()

    # Test forward train with an empty truth batch
    mm_inputs = _demo_mm_inputs(input_shape, num_items=[0])
    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')
    img_metas[0]['is_video_data'] = True
    gt_bboxes = mm_inputs['gt_bboxes']
    gt_labels = mm_inputs['gt_labels']
    gt_masks = mm_inputs['gt_masks']
    losses = detector.forward(
        img=imgs,
        img_metas=img_metas,
        gt_bboxes=gt_bboxes,
        gt_labels=gt_labels,
        ref_img=imgs.clone()[:, None, ...],
        ref_img_metas=copy.deepcopy(img_metas),
        ref_gt_bboxes=copy.deepcopy(gt_bboxes),
        ref_gt_labels=copy.deepcopy(gt_labels),
        gt_masks=gt_masks,
        ref_gt_masks=copy.deepcopy(gt_masks),
        return_loss=True)
    assert isinstance(losses, dict)
    loss, _ = detector._parse_losses(losses)
    loss.requires_grad_(True)
    assert float(loss.item()) > 0
    loss.backward()

    # Test forward test
    with torch.no_grad():
        imgs = torch.cat([imgs, imgs.clone()], dim=0)
        img_list = [g[None, :] for g in imgs]
        img_metas.extend(copy.deepcopy(img_metas))
        for i in range(1, len(img_metas)):
            img_metas[i]['frame_id'] = i
        results = defaultdict(list)
        for one_img, one_meta in zip(img_list, img_metas):
            result = detector.forward([one_img], [[one_meta]],
                                      return_loss=False)
            for k, v in result.items():
                results[k].append(v)


@pytest.mark.parametrize(
    'cfg_file', ['mot/qdtrack/qdtrack_faster-rcnn_fpn_12e_bdd100k.py'])
def test_mot_forward(cfg_file):
    config = _get_config_module(cfg_file)
    model = copy.deepcopy(config.model)
    model.detector.pretrained = None
    from mmtrack.models import build_model
    tracker = build_model(model)

    input_shape = (1, 3, 256, 256)

    # Test forward train with a non-empty truth batch
    mm_inputs = _demo_mm_inputs(input_shape, num_items=[2], with_track=True)
    img = mm_inputs['imgs']
    img_metas = mm_inputs['img_metas']
    gt_bboxes = mm_inputs['gt_bboxes']
    gt_labels = mm_inputs['gt_labels']
    gt_match_indices = mm_inputs['gt_match_indices']
    losses = tracker.forward(
        img,
        img_metas,
        gt_bboxes=gt_bboxes,
        gt_labels=gt_labels,
        gt_match_indices=gt_match_indices,
        ref_img=copy.deepcopy(img),
        ref_img_metas=copy.deepcopy(img_metas),
        ref_gt_bboxes=copy.deepcopy(gt_bboxes),
        ref_gt_labels=copy.deepcopy(gt_labels),
        ref_gt_match_indices=copy.deepcopy(gt_match_indices),
        gt_bboxes_ignore=None,
        gt_masks=None,
        ref_gt_bboxes_ignore=None,
        ref_gt_masks=None)
    assert isinstance(losses, dict)
    loss, _ = tracker._parse_losses(losses)
    loss.requires_grad_(True)
    assert float(loss.item()) > 0
    loss.backward()

    # Test forward train with an empty truth batch
    mm_inputs = _demo_mm_inputs(input_shape, num_items=[0], with_track=True)
    img = mm_inputs['imgs']
    img_metas = mm_inputs['img_metas']
    gt_bboxes = mm_inputs['gt_bboxes']
    gt_labels = mm_inputs['gt_labels']
    gt_match_indices = mm_inputs['gt_match_indices']
    losses = tracker.forward(
        img,
        img_metas,
        gt_bboxes=gt_bboxes,
        gt_labels=gt_labels,
        gt_match_indices=gt_match_indices,
        ref_img=copy.deepcopy(img),
        ref_img_metas=copy.deepcopy(img_metas),
        ref_gt_bboxes=copy.deepcopy(gt_bboxes),
        ref_gt_labels=copy.deepcopy(gt_labels),
        ref_gt_match_indices=copy.deepcopy(gt_match_indices),
        gt_bboxes_ignore=None,
        gt_masks=None,
        ref_gt_bboxes_ignore=None,
        ref_gt_masks=None)
    assert isinstance(losses, dict)
    loss, _ = tracker._parse_losses(losses)
    loss.requires_grad_(True)
    assert float(loss.item()) > 0
    loss.backward()

    # Test forward test
    with torch.no_grad():
        N = 5
        img_list = [img for i in range(N)]
        img_metas = [img_metas for i in range(N)]
        for i in range(1, len(img_metas)):
            img_metas[i][0]['frame_id'] = i
        results = defaultdict(list)
        for _img, _meta in zip(img_list, img_metas):
            result = tracker.forward([_img], [_meta], return_loss=False)
            for k, v in result.items():
                results[k].append(v)
        assert 'bbox_results' in results
        assert 'track_results' in results


def _demo_mm_inputs(
        input_shape=(1, 3, 300, 300),
        num_items=None,
        num_classes=10,
        with_track=False):
    """Create a superset of inputs needed to run test or train batches.

    Args:
        input_shape (tuple):
            input batch dimensions

        num_items (None | List[int]):
            specifies the number of boxes in each batch item

        num_classes (int):
            number of different labels a box might have
    """
    from mmdet.core import BitmapMasks

    (N, C, H, W) = input_shape

    rng = np.random.RandomState(0)

    imgs = rng.rand(*input_shape)

    img_metas = [{
        'img_shape': (H, W, C),
        'ori_shape': (H, W, C),
        'pad_shape': (H, W, C),
        'filename': '<demo>.png',
        'scale_factor': 1.0,
        'flip': False,
        'frame_id': 0,
        'img_norm_cfg': {
            'mean': (128.0, 128.0, 128.0),
            'std': (10.0, 10.0, 10.0)
        }
    } for i in range(N)]

    gt_bboxes = []
    gt_labels = []
    gt_masks = []
    gt_match_indices = []

    for batch_idx in range(N):
        if num_items is None:
            num_boxes = rng.randint(1, 10)
        else:
            num_boxes = num_items[batch_idx]

        cx, cy, bw, bh = rng.rand(num_boxes, 4).T

        tl_x = ((cx * W) - (W * bw / 2)).clip(0, W)
        tl_y = ((cy * H) - (H * bh / 2)).clip(0, H)
        br_x = ((cx * W) + (W * bw / 2)).clip(0, W)
        br_y = ((cy * H) + (H * bh / 2)).clip(0, H)

        boxes = np.vstack([tl_x, tl_y, br_x, br_y]).T
        class_idxs = rng.randint(1, num_classes, size=num_boxes)

        gt_bboxes.append(torch.FloatTensor(boxes))
        gt_labels.append(torch.LongTensor(class_idxs))
        if with_track:
            gt_match_indices.append(torch.arange(boxes.shape[0]))

    mask = np.random.randint(0, 2, (len(boxes), H, W), dtype=np.uint8)
    gt_masks.append(BitmapMasks(mask, H, W))

    mm_inputs = {
        'imgs': torch.FloatTensor(imgs).requires_grad_(True),
        'img_metas': img_metas,
        'gt_bboxes': gt_bboxes,
        'gt_labels': gt_labels,
        'gt_bboxes_ignore': None,
        'gt_masks': gt_masks,
    }
    if with_track:
        mm_inputs['gt_match_indices'] = gt_match_indices
    return mm_inputs
