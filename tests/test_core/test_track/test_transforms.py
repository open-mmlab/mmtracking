# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmdet.core.bbox.demodata import random_boxes


def test_imrenormalize():
    from mmtrack.core import imrenormalize
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True)
    new_img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True)

    img = np.random.randn(128, 256, 3).astype(np.float32)
    new_img = imrenormalize(img, img_norm_cfg, new_img_norm_cfg)
    assert isinstance(new_img, np.ndarray)
    assert new_img.shape == (128, 256, 3)
    assert np.allclose(img, new_img, atol=1e-6)

    img = torch.randn(1, 3, 128, 256, dtype=torch.float)
    new_img = imrenormalize(img, img_norm_cfg, new_img_norm_cfg)
    assert isinstance(new_img, torch.Tensor)
    assert new_img.shape == (1, 3, 128, 256)
    assert np.allclose(img, new_img, atol=1e-6)


def test_track2result():
    from mmtrack.core import track2result, track2result_with_segm

    # pseudo data
    num_objects, num_classes = 8, 4
    bboxes = random_boxes(num_objects, 640)
    scores = torch.FloatTensor(num_objects, 1).uniform_(0, 1)
    bboxes = torch.cat([bboxes, scores], dim=1)
    # leave the results of the last class as empty
    labels = torch.randint(0, num_classes - 1, (num_objects, ))
    ids = torch.arange(num_objects)
    masks = torch.randint(0, 2, (num_objects, 100, 100)).bool()

    # test track2result
    result = track2result(bboxes, labels, ids, num_classes)

    assert len(result) == num_classes
    assert result[-1].shape == (0, 6)
    assert isinstance(result[0], np.ndarray)
    for i in range(num_classes):
        assert result[i].shape[0] == (labels == i).sum()
        assert result[i].shape[1] == 6

    # test track2result_with_segm
    track_bbox_result, track_segm_result = track2result_with_segm(
        bboxes, labels, masks, ids, num_classes)

    assert len(track_segm_result) == num_classes
    assert len(track_segm_result[-1]) == 0
    assert isinstance(track_segm_result[-1], list)
    for i in range(num_classes):
        assert len(track_segm_result[i]) == (labels == i).sum()


def test_restore_result():
    from mmtrack.core import restore_result, restore_result_with_segm
    num_classes = 3
    num_objects = [2, 0, 2]

    result = [np.random.randn(num_objects[i], 5) for i in range(num_classes)]
    bboxes, labels = restore_result(result, return_ids=False)
    assert bboxes.shape == (4, 5)
    assert (labels == np.array([0, 0, 2, 2])).all()
    segm_results = [[] for i in range(num_classes)]
    for i in range(bboxes.shape[0]):
        segm_results[labels[i]].append(np.random.randint(0, 2, (100, 100)))
    bboxes, labels, masks = restore_result_with_segm(
        result, segm_results, 100, 100, return_ids=False)
    assert masks.shape == (4, 100, 100)

    result = [np.random.randn(num_objects[i], 6) for i in range(num_classes)]
    bboxes, labels, ids = restore_result(result, return_ids=True)
    assert bboxes.shape == (4, 5)
    assert (labels == np.array([0, 0, 2, 2])).all()
    assert len(ids) == 4
    segm_results = [[] for i in range(num_classes)]
    for i in range(bboxes.shape[0]):
        segm_results[labels[i]].append(np.random.randint(0, 2, (100, 100)))
    bboxes, labels, masks, ids = restore_result_with_segm(
        result, segm_results, 100, 100, return_ids=True)
    assert masks.shape == (4, 100, 100)
