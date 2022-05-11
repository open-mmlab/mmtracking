# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np


def test_interpolate_tracks():
    from mmtrack.core import interpolate_tracks
    frame_id = np.arange(100) // 10
    tracklet_id = np.random.randint(low=1, high=5, size=(100))
    bboxes = np.random.random((100, 4)) * 100
    scores = np.random.random((100)) * 100
    in_results = np.concatenate(
        (frame_id[:, None], tracklet_id[:, None], bboxes, scores[:, None]),
        axis=1)
    out_results = interpolate_tracks(in_results)
    assert out_results.shape[1] == in_results.shape[1]
    # the range of frame ids should not change
    assert min(out_results[:, 0]) == min(in_results[:, 0])
    assert max(out_results[:, 0]) == max(in_results[:, 0])
    # the range of track ids should not change
    assert min(out_results[:, 1]) == min(in_results[:, 1])
    assert max(out_results[:, 1]) == max(in_results[:, 1])
