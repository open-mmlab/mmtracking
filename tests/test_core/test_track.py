import numpy as np
import pytest
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


def test_compute_distance():
    from mmtrack.core import compute_distance_matrix
    input1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
    input2 = np.array([[4, 3], [2, 1]], dtype=np.float32)

    distance = compute_distance_matrix(input1, input2, metric='euclidean')
    assert np.allclose(distance[0, 0], np.sqrt(np.array(10.)))
    assert np.allclose(distance[0, 1], np.sqrt(np.array(2.)))
    assert np.allclose(distance[1, 0], np.sqrt(np.array(2.)))
    assert np.allclose(distance[1, 1], np.sqrt(np.array(10.)))

    distance = compute_distance_matrix(
        input1, input2, metric='squared_euclidean')
    assert np.allclose(distance[0, 0], np.array(10.))
    assert np.allclose(distance[0, 1], np.array(2.))
    assert np.allclose(distance[1, 0], np.array(2.))
    assert np.allclose(distance[1, 1], np.array(10.))

    distance = compute_distance_matrix(input1, input2, metric='cosine')
    assert np.allclose(distance[0, 0],
                       1 - np.array(2.) / np.sqrt(np.array(5.)))
    assert np.allclose(distance[0, 1], 1 - np.array(4.) / np.array(5.))
    assert np.allclose(distance[1, 0], 1 - np.array(24.) / np.array(25.))
    assert np.allclose(distance[1, 1],
                       1 - np.array(2.) / np.sqrt(np.array(5.)))

    input1 = torch.from_numpy(input1)
    input2 = torch.from_numpy(input2)

    distance = compute_distance_matrix(input1, input2, metric='euclidean')
    assert torch.allclose(distance[0, 0], torch.sqrt(torch.Tensor([10.])))
    assert torch.allclose(distance[0, 1], torch.sqrt(torch.Tensor([2.])))
    assert torch.allclose(distance[1, 0], torch.sqrt(torch.Tensor([2.])))
    assert torch.allclose(distance[1, 1], torch.sqrt(torch.Tensor([10.])))

    distance = compute_distance_matrix(
        input1, input2, metric='squared_euclidean')
    assert torch.allclose(distance[0, 0], torch.Tensor([10.]))
    assert torch.allclose(distance[0, 1], torch.Tensor([2.]))
    assert torch.allclose(distance[1, 0], torch.Tensor([2.]))
    assert torch.allclose(distance[1, 1], torch.Tensor([10.]))

    distance = compute_distance_matrix(input1, input2, metric='cosine')
    assert torch.allclose(
        distance[0, 0],
        1 - torch.Tensor([2.]) / torch.sqrt(torch.Tensor([5.])))
    assert torch.allclose(distance[0, 1],
                          1 - torch.Tensor([4.]) / torch.Tensor([5.]))
    assert torch.allclose(distance[1, 0],
                          1 - torch.Tensor([24.]) / torch.Tensor([25.]))
    assert torch.allclose(
        distance[1, 1],
        1 - torch.Tensor([2.]) / torch.sqrt(torch.Tensor([5.])))

    with pytest.raises(AssertionError):
        distance = compute_distance_matrix(
            input1, input2, metric='mahalanobis')

    input1 = torch.Tensor([1, 2, 3])
    with pytest.raises(AssertionError):
        distance = compute_distance_matrix(input1, input2, metric='euclidean')

    input1 = torch.Tensor([[1, 2, 3], [2, 3, 4]])
    with pytest.raises(AssertionError):
        distance = compute_distance_matrix(input1, input2, metric='euclidean')

    input1 = [[1, 2], [2, 3]]
    with pytest.raises(TypeError):
        distance = compute_distance_matrix(input1, input2, metric='euclidean')
