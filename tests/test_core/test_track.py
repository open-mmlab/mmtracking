import numpy as np
import torch
from mmdet.core.bbox.demodata import random_boxes


def test_track2result():
    from mmtrack.core import track2result

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
    from mmtrack.core import restore_result
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


def test_embed_similarity():
    from mmtrack.core import embed_similarity
    key_embeds = torch.randn(20, 256)
    ref_embeds = torch.randn(10, 256)

    sims = embed_similarity(
        key_embeds,
        ref_embeds,
        method='dot_product',
        temperature=-1,
        transpose=True)
    assert sims.size() == (20, 10)

    sims = embed_similarity(
        key_embeds,
        ref_embeds.t(),
        method='dot_product',
        temperature=-1,
        transpose=False)
    assert sims.size() == (20, 10)

    sims = embed_similarity(
        key_embeds,
        ref_embeds,
        method='dot_product',
        temperature=0.07,
        transpose=True)
    assert sims.size() == (20, 10)

    sims = embed_similarity(
        key_embeds,
        ref_embeds,
        method='cosine',
        temperature=-1,
        transpose=True)
    assert sims.size() == (20, 10)
    assert sims.max() <= 1
