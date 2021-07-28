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
                         ['sot/siamese_rpn/siamese_rpn_r50_1x_lasot.py'])
def test_sot_forward(cfg_file):
    config = _get_config_module(cfg_file)
    model = copy.deepcopy(config.model)
    model.pretrains = None

    from mmtrack.models import build_model
    tracktor = build_model(model)

    # Test forward train with a non-empty truth batch
    input_shape = (1, 3, 127, 127)
    mm_inputs = _demo_mm_inputs(input_shape, num_items=[1])
    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')
    gt_bboxes = mm_inputs['gt_bboxes']

    search_input_shape = (1, 3, 255, 255)
    search_mm_inputs = _demo_mm_inputs(search_input_shape, num_items=[1])
    search_img = search_mm_inputs.pop('imgs')[None]
    search_img_metas = search_mm_inputs.pop('img_metas')
    search_gt_bboxes = search_mm_inputs['gt_bboxes']
    img_inds = search_gt_bboxes[0].new_full((search_gt_bboxes[0].size(0), 1),
                                            0)
    search_gt_bboxes[0] = torch.cat((img_inds, search_gt_bboxes[0]), dim=1)

    losses = tracktor.forward(
        img=imgs,
        img_metas=img_metas,
        gt_bboxes=gt_bboxes,
        search_img=search_img,
        search_img_metas=search_img_metas,
        search_gt_bboxes=search_gt_bboxes,
        is_positive_pairs=[True],
        return_loss=True)
    assert isinstance(losses, dict)
    loss, _ = tracktor._parse_losses(losses)
    loss.requires_grad_(True)
    assert float(loss.item()) > 0
    loss.backward()

    losses = tracktor.forward(
        img=imgs,
        img_metas=img_metas,
        gt_bboxes=gt_bboxes,
        search_img=search_img,
        search_img_metas=search_img_metas,
        search_gt_bboxes=search_gt_bboxes,
        is_positive_pairs=[False],
        return_loss=True)
    assert isinstance(losses, dict)
    loss, _ = tracktor._parse_losses(losses)
    loss.requires_grad_(True)
    assert float(loss.item()) > 0
    loss.backward()

    # Test forward test
    with torch.no_grad():
        imgs = torch.cat([imgs, imgs.clone()], dim=0)
        img_list = [g[None, :] for g in imgs]
        img_metas.extend(copy.deepcopy(img_metas))
        for i in range(len(img_metas)):
            img_metas[i]['frame_id'] = i
        gt_bboxes.extend(copy.deepcopy(gt_bboxes))
        results = defaultdict(list)
        for one_img, one_meta, one_gt_bboxes in zip(img_list, img_metas,
                                                    gt_bboxes):
            result = tracktor.forward([one_img], [[one_meta]],
                                      gt_bboxes=[one_gt_bboxes],
                                      return_loss=False)
            for k, v in result.items():
                results[k].append(v)


@pytest.mark.parametrize(
    'cfg_file', ['vid/dff/dff_faster_rcnn_r101_dc5_1x_imagenetvid.py'])
def test_vid_dff_style_forward(cfg_file):
    config = _get_config_module(cfg_file)
    model = copy.deepcopy(config.model)
    model.pretrains = None
    model.detector.pretrained = None

    from mmtrack.models import build_model
    detector = build_model(model)

    # Test forward train with a non-empty truth batch
    input_shape = (1, 3, 256, 256)
    mm_inputs = _demo_mm_inputs(input_shape, num_items=[10])
    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')
    img_metas[0]['is_video_data'] = True
    gt_bboxes = mm_inputs['gt_bboxes']
    gt_labels = mm_inputs['gt_labels']
    gt_masks = mm_inputs['gt_masks']

    ref_input_shape = (1, 3, 256, 256)
    ref_mm_inputs = _demo_mm_inputs(ref_input_shape, num_items=[11])
    ref_img = ref_mm_inputs.pop('imgs')[None]
    ref_img_metas = ref_mm_inputs.pop('img_metas')
    ref_img_metas[0]['is_video_data'] = True
    ref_gt_bboxes = ref_mm_inputs['gt_bboxes']
    ref_gt_labels = ref_mm_inputs['gt_labels']
    ref_gt_masks = ref_mm_inputs['gt_masks']

    losses = detector.forward(
        img=imgs,
        img_metas=img_metas,
        gt_bboxes=gt_bboxes,
        gt_labels=gt_labels,
        ref_img=ref_img,
        ref_img_metas=ref_img_metas,
        ref_gt_bboxes=ref_gt_bboxes,
        ref_gt_labels=ref_gt_labels,
        gt_masks=gt_masks,
        ref_gt_masks=ref_gt_masks,
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

    ref_input_shape = (1, 3, 256, 256)
    ref_mm_inputs = _demo_mm_inputs(ref_input_shape, num_items=[0])
    ref_img = ref_mm_inputs.pop('imgs')[None]
    ref_img_metas = ref_mm_inputs.pop('img_metas')
    ref_img_metas[0]['is_video_data'] = True
    ref_gt_bboxes = ref_mm_inputs['gt_bboxes']
    ref_gt_labels = ref_mm_inputs['gt_labels']
    ref_gt_masks = ref_mm_inputs['gt_masks']

    losses = detector.forward(
        img=imgs,
        img_metas=img_metas,
        gt_bboxes=gt_bboxes,
        gt_labels=gt_labels,
        ref_img=ref_img,
        ref_img_metas=ref_img_metas,
        ref_gt_bboxes=ref_gt_bboxes,
        ref_gt_labels=ref_gt_labels,
        gt_masks=gt_masks,
        ref_gt_masks=ref_gt_masks,
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
        for i in range(len(img_metas)):
            img_metas[i]['frame_id'] = i
        results = defaultdict(list)
        for one_img, one_meta in zip(img_list, img_metas):
            result = detector.forward([one_img], [[one_meta]],
                                      return_loss=False)
            for k, v in result.items():
                results[k].append(v)


@pytest.mark.parametrize('cfg_file', [
    'vid/fgfa/fgfa_faster_rcnn_r101_dc5_1x_imagenetvid.py',
    'vid/selsa/selsa_faster_rcnn_r101_dc5_1x_imagenetvid.py'
])
def test_vid_fgfa_style_forward(cfg_file):
    config = _get_config_module(cfg_file)
    model = copy.deepcopy(config.model)
    model.pretrains = None
    model.detector.pretrained = None

    from mmtrack.models import build_model
    detector = build_model(model)

    # Test forward train with a non-empty truth batch
    input_shape = (1, 3, 256, 256)
    mm_inputs = _demo_mm_inputs(input_shape, num_items=[10])
    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')
    img_metas[0]['is_video_data'] = True
    gt_bboxes = mm_inputs['gt_bboxes']
    gt_labels = mm_inputs['gt_labels']
    gt_masks = mm_inputs['gt_masks']

    ref_input_shape = (2, 3, 256, 256)
    ref_mm_inputs = _demo_mm_inputs(ref_input_shape, num_items=[9, 11])
    ref_img = ref_mm_inputs.pop('imgs')[None]
    ref_img_metas = ref_mm_inputs.pop('img_metas')
    ref_img_metas[0]['is_video_data'] = True
    ref_img_metas[1]['is_video_data'] = True
    ref_gt_bboxes = ref_mm_inputs['gt_bboxes']
    ref_gt_labels = ref_mm_inputs['gt_labels']
    ref_gt_masks = ref_mm_inputs['gt_masks']

    losses = detector.forward(
        img=imgs,
        img_metas=img_metas,
        gt_bboxes=gt_bboxes,
        gt_labels=gt_labels,
        ref_img=ref_img,
        ref_img_metas=[ref_img_metas],
        ref_gt_bboxes=ref_gt_bboxes,
        ref_gt_labels=ref_gt_labels,
        gt_masks=gt_masks,
        ref_gt_masks=ref_gt_masks,
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

    ref_mm_inputs = _demo_mm_inputs(ref_input_shape, num_items=[0, 0])
    ref_imgs = ref_mm_inputs.pop('imgs')[None]
    ref_img_metas = ref_mm_inputs.pop('img_metas')
    ref_img_metas[0]['is_video_data'] = True
    ref_img_metas[1]['is_video_data'] = True
    ref_gt_bboxes = ref_mm_inputs['gt_bboxes']
    ref_gt_labels = ref_mm_inputs['gt_labels']
    ref_gt_masks = ref_mm_inputs['gt_masks']

    losses = detector.forward(
        img=imgs,
        img_metas=img_metas,
        gt_bboxes=gt_bboxes,
        gt_labels=gt_labels,
        ref_img=ref_imgs,
        ref_img_metas=[ref_img_metas],
        ref_gt_bboxes=ref_gt_bboxes,
        ref_gt_labels=ref_gt_labels,
        gt_masks=gt_masks,
        ref_gt_masks=ref_gt_masks,
        return_loss=True)
    assert isinstance(losses, dict)
    loss, _ = detector._parse_losses(losses)
    loss.requires_grad_(True)
    assert float(loss.item()) > 0
    loss.backward()

    # Test forward test with frame_stride=1 and frame_range=[-1,0]
    with torch.no_grad():
        imgs = torch.cat([imgs, imgs.clone()], dim=0)
        img_list = [g[None, :] for g in imgs]
        img_metas.extend(copy.deepcopy(img_metas))
        for i in range(len(img_metas)):
            img_metas[i]['frame_id'] = i
            img_metas[i]['num_left_ref_imgs'] = 1
            img_metas[i]['frame_stride'] = 1
        ref_imgs = [ref_imgs.clone(), imgs[[0]][None].clone()]
        ref_img_metas = [
            copy.deepcopy(ref_img_metas),
            copy.deepcopy([img_metas[0]])
        ]
        results = defaultdict(list)
        for one_img, one_meta, ref_img, ref_img_meta in zip(
                img_list, img_metas, ref_imgs, ref_img_metas):
            result = detector.forward([one_img], [[one_meta]],
                                      ref_img=[ref_img],
                                      ref_img_metas=[[ref_img_meta]],
                                      return_loss=False)
            for k, v in result.items():
                results[k].append(v)


@pytest.mark.parametrize('cfg_file', [
    'mot/deepsort/deepsort_faster-rcnn_fpn_4e_mot17-private-half.py',
    'mot/tracktor/tracktor_faster-rcnn_r50_fpn_8e_mot20-public.py'
])
def test_mot_tracker_forward(cfg_file):
    config = _get_config_module(cfg_file)
    model = copy.deepcopy(config.model)
    model.pretrains = None
    model.detector.pretrained = None

    from mmtrack.models import build_model
    mot = build_model(model)
    mot.eval()

    input_shape = (1, 3, 256, 256)
    mm_inputs = _demo_mm_inputs(input_shape, num_items=[10], with_track=True)
    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')
    with torch.no_grad():
        imgs = torch.cat([imgs, imgs.clone()], dim=0)
        img_list = [g[None, :] for g in imgs]
        img2_metas = copy.deepcopy(img_metas)
        img2_metas[0]['frame_id'] = 1
        img_metas.extend(img2_metas)
        results = defaultdict(list)
        for one_img, one_meta in zip(img_list, img_metas):
            result = mot.forward([one_img], [[one_meta]], return_loss=False)
            for k, v in result.items():
                results[k].append(v)


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
