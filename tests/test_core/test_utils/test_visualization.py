# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import tempfile

import numpy as np
import pytest

from mmtrack.core.utils import visualization as vis


def test_imshow_mot_errors():
    # only support opencv and matplotlib
    with pytest.raises(NotImplementedError):
        vis.imshow_mot_errors(backend='pillow')


def test_cv2_show_wrong_tracks():
    tmp_filename = osp.join(tempfile.gettempdir(), 'mot_error_image',
                            'image.jpg')
    image = np.ones((100, 100, 3), np.uint8)
    bboxes = np.array([[20, 10, 30, 30, 0.5], [30, 40, 60, 60, 0.5]])
    ids = np.array([0, 1])
    error_types = np.array([0, 1])
    out_image = vis._cv2_show_wrong_tracks(
        image, bboxes, ids, error_types, out_file=tmp_filename, show=False)
    assert osp.isfile(tmp_filename)
    assert image.shape == out_image.shape
    os.remove(tmp_filename)

    # not support gray image
    with pytest.raises(AssertionError):
        image = np.ones((100, 100), np.uint8)
        bboxes = np.array([[20, 10, 30, 30, 0.5], [30, 40, 60, 60, 0.5]])
        ids = np.array([0, 1])
        error_types = np.array([0, 1])
        out_image = vis._cv2_show_wrong_tracks(
            image, bboxes, ids, error_types, out_file=None, show=False)

    # bboxes.ndim should be 2
    with pytest.raises(AssertionError):
        image = np.ones((100, 100, 3), np.uint8)
        bboxes = np.array([30, 40, 60, 60, 0.5])
        ids = np.array([0, 1])
        error_types = np.array([0, 1])
        out_image = vis._cv2_show_wrong_tracks(
            image, bboxes, ids, error_types, out_file=None, show=False)

    # ids.ndim should be 1
    with pytest.raises(AssertionError):
        image = np.ones((100, 100, 3), np.uint8)
        bboxes = np.array([[20, 10, 30, 30, 0.5], [30, 40, 60, 60, 0.5]])
        ids = np.array([[0], [1]])
        error_types = np.array([0, 1])
        out_image = vis._cv2_show_wrong_tracks(
            image, bboxes, ids, error_types, out_file=None, show=False)

    # error_types.ndim should be 1
    with pytest.raises(AssertionError):
        image = np.ones((100, 100, 3), np.uint8)
        bboxes = np.array([[20, 10, 30, 30, 0.5], [30, 40, 60, 60, 0.5]])
        ids = np.array([0, 1])
        error_types = np.array([[0], [1]])
        out_image = vis._plt_show_wrong_tracks(
            image, bboxes, ids, error_types, out_file=None, show=False)

    # bboxes.shape[0] and ids.shape[0] should have the same length
    with pytest.raises(AssertionError):
        image = np.ones((100, 100, 3), np.uint8)
        bboxes = np.array([[20, 10, 30, 30, 0.5], [30, 40, 60, 60, 0.5]])
        ids = np.array([0, 1, 2])
        error_types = np.array([0, 1])
        out_image = vis._cv2_show_wrong_tracks(
            image, bboxes, ids, error_types, out_file=None, show=False)

    # bboxes.shape[0] should have 5
    with pytest.raises(AssertionError):
        image = np.ones((100, 100, 3), np.uint8)
        bboxes = np.array([[20, 10, 30, 30], [30, 40, 60, 60]])
        ids = np.array([0, 1])
        error_types = np.array([0, 1])
        out_image = vis._cv2_show_wrong_tracks(
            image, bboxes, ids, error_types, out_file=None, show=False)


def test_plt_show_wrong_tracks():
    tmp_filename = osp.join(tempfile.gettempdir(), 'mot_error_image',
                            'image.jpg')
    image = np.ones((100, 100, 3), np.uint8)
    bboxes = np.array([[20, 10, 30, 30, 0.5], [30, 40, 60, 60, 0.5]])
    ids = np.array([0, 1])
    error_types = np.array([0, 1])
    out_image = vis._plt_show_wrong_tracks(
        image, bboxes, ids, error_types, out_file=tmp_filename, show=False)
    assert osp.isfile(tmp_filename)
    assert image.shape == out_image.shape
    os.remove(tmp_filename)

    # not support gray image
    with pytest.raises(AssertionError):
        image = np.ones((100, 100), np.uint8)
        bboxes = np.array([[20, 10, 30, 30, 0.5], [30, 40, 60, 60, 0.5]])
        ids = np.array([0, 1])
        error_types = np.array([0, 1])
        out_image = vis._plt_show_wrong_tracks(
            image, bboxes, ids, error_types, out_file=None, show=False)

    # bboxes.ndim should be 2
    with pytest.raises(AssertionError):
        image = np.ones((100, 100, 3), np.uint8)
        bboxes = np.array([30, 40, 60, 60, 0.5])
        ids = np.array([0, 1])
        error_types = np.array([0, 1])
        out_image = vis._plt_show_wrong_tracks(
            image, bboxes, ids, error_types, out_file=None, show=False)

    # ids.ndim should be 1
    with pytest.raises(AssertionError):
        image = np.ones((100, 100, 3), np.uint8)
        bboxes = np.array([[20, 10, 30, 30, 0.5], [30, 40, 60, 60, 0.5]])
        ids = np.array([[0], [1]])
        error_types = np.array([0, 1])
        out_image = vis._plt_show_wrong_tracks(
            image, bboxes, ids, error_types, out_file=None, show=False)

    # error_types.ndim should be 1
    with pytest.raises(AssertionError):
        image = np.ones((100, 100, 3), np.uint8)
        bboxes = np.array([[20, 10, 30, 30, 0.5], [30, 40, 60, 60, 0.5]])
        ids = np.array([0, 1])
        error_types = np.array([[0], [1]])
        out_image = vis._plt_show_wrong_tracks(
            image, bboxes, ids, error_types, out_file=None, show=False)

    # bboxes.shape[0] and ids.shape[0] should have the same length
    with pytest.raises(AssertionError):
        image = np.ones((100, 100, 3), np.uint8)
        bboxes = np.array([[20, 10, 30, 30, 0.5], [30, 40, 60, 60, 0.5]])
        ids = np.array([0, 1, 2])
        error_types = np.array([0, 1])
        out_image = vis._plt_show_wrong_tracks(
            image, bboxes, ids, error_types, out_file=None, show=False)

    # bboxes.shape[0] should have 5
    with pytest.raises(AssertionError):
        image = np.ones((100, 100, 3), np.uint8)
        bboxes = np.array([[20, 10, 30, 30], [30, 40, 60, 60, 0.5]])
        ids = np.array([0, 1])
        error_types = np.array([0, 1])
        out_image = vis._plt_show_wrong_tracks(
            image, bboxes, ids, error_types, out_file=None, show=False)
