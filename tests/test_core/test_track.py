import numpy as np
import torch
from mmdet.core.bbox.demodata import random_boxes

from mmtrack.core import restore_result, track2result


def test_track2result():
    # pseudo data
    num_objects, num_classes = 8, 4
    bboxes = random_boxes(num_objects, 640)
    scores = torch.FloatTensor(num_objects, 1).uniform_(0, 1)
    bboxes = torch.cat([bboxes, scores], dim=1)
    # leave the results of the last class as empty
    labels = torch.randint(0, num_classes - 1, (num_objects, ))
    ids = torch.arange(num_objects)

    # run
    result = track2result(bboxes, labels, ids, num_classes)

    # test
    assert len(result) == num_classes
    assert result[-1].shape == (0, 6)
    assert isinstance(result[0], np.ndarray)
    for i in range(num_classes):
        assert result[i].shape[0] == (labels == i).sum()
        assert result[i].shape[1] == 6


def test_restore_result():
    num_classes = 3
    num_objects = [2, 0, 2]

    result = [np.random.randn(num_objects[i], 5) for i in range(num_classes)]
    bboxes, labels = restore_result(result, return_ids=False)
    assert bboxes.shape == (4, 5)
    assert (labels == np.array([0, 0, 2, 2])).all()

    result = [np.random.randn(num_objects[i], 6) for i in range(num_classes)]
    bboxes, labels, ids = restore_result(result, return_ids=True)
    assert bboxes.shape == (4, 5)
    assert (labels == np.array([0, 0, 2, 2])).all()
    assert len(ids) == 4
