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


def test_outs2results():
    from mmtrack.core import outs2results

    # pseudo data
    num_objects, num_classes, image_size = 8, 4, 100
    bboxes = random_boxes(num_objects, image_size)
    scores = torch.FloatTensor(num_objects, 1).uniform_(0, 1)
    bboxes = torch.cat([bboxes, scores], dim=1)
    # leave the results of the last class as empty
    labels = torch.randint(0, num_classes - 1, (num_objects, ))
    ids = torch.arange(num_objects)
    masks = torch.randint(0, 2, (num_objects, image_size, image_size)).bool()

    # test track2result without ids
    outs = dict(
        bboxes=bboxes, labels=labels, masks=masks, num_classes=num_classes)
    results = outs2results(outs)

    for key in ['bboxes', 'masks']:
        assert key in results
    assert len(results['bboxes']) == num_classes
    assert isinstance(results['bboxes'][0], np.ndarray)
    assert results['bboxes'][-1].shape == (0, 5)
    assert len(results['masks']) == num_classes
    assert isinstance(results['masks'][-1], list)
    assert len(results['masks'][-1]) == 0
    for i in range(num_classes):
        assert results['bboxes'][i].shape[0] == (labels == i).sum()
        assert results['bboxes'][i].shape[1] == 5
        assert len(results['masks'][i]) == (labels == i).sum()
        if len(results['masks'][i]) > 0:
            assert results['masks'][i][0].shape == (image_size, image_size)

    # test track2result with ids
    outs['ids'] = ids
    results = outs2results(outs)

    for key in ['bboxes', 'masks']:
        assert key in results
    assert len(results['bboxes']) == num_classes
    assert isinstance(results['bboxes'][0], np.ndarray)
    assert results['bboxes'][-1].shape == (0, 6)
    assert len(results['masks']) == num_classes
    assert isinstance(results['masks'][-1], list)
    assert len(results['masks'][-1]) == 0
    for i in range(num_classes):
        assert results['bboxes'][i].shape[0] == (labels == i).sum()
        assert results['bboxes'][i].shape[1] == 6
        assert len(results['masks'][i]) == (labels == i).sum()
        if len(results['masks'][i]) > 0:
            assert results['masks'][i][0].shape == (image_size, image_size)


def test_results2outs():
    from mmtrack.core import results2outs
    num_classes = 3
    num_objects = [2, 0, 2]
    gt_labels = []
    for id, num in enumerate(num_objects):
        gt_labels.extend([id for _ in range(num)])
    image_size = 100

    bbox_result = [
        np.random.randint(low=0, high=image_size, size=(num_objects[i], 5))
        for i in range(num_classes)
    ]
    bbox_result_with_ids = [
        np.random.randint(low=0, high=image_size, size=(num_objects[i], 6))
        for i in range(num_classes)
    ]
    mask_results = [[] for i in range(num_classes)]
    for cls_id in range(num_classes):
        for obj_id in range(num_objects[cls_id]):
            mask_results[cls_id].append(
                np.random.randint(0, 2, (image_size, image_size)))

    # test results2outs without ids
    results = dict(
        bboxes=bbox_result,
        masks=mask_results,
        mask_shape=(image_size, image_size))
    outs = results2outs(results)

    for key in ['bboxes', 'labels', 'masks']:
        assert key in outs
    assert outs['bboxes'].shape == (sum(num_objects), 5)
    assert (outs['labels'] == np.array(gt_labels)).all()
    assert outs['masks'].shape == (sum(num_objects), image_size, image_size)

    # test results2outs with ids
    results = dict(
        bboxes=bbox_result_with_ids,
        masks=mask_results,
        mask_shape=(image_size, image_size))
    outs = results2outs(results)

    for key in ['bboxes', 'labels', 'ids', 'masks']:
        assert key in outs
    assert outs['bboxes'].shape == (sum(num_objects), 5)
    assert (outs['labels'] == np.array(gt_labels)).all()
    assert outs['ids'].shape == (sum(num_objects), )
    assert outs['masks'].shape == (sum(num_objects), image_size, image_size)
