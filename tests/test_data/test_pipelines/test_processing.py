# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmcv.utils import build_from_cfg

from mmtrack.datasets import PIPELINES


def test_trident_sampling():
    process = dict(
        type='TridentSampling',
        num_search_frames=1,
        num_template_frames=2,
        max_frame_range=[200],
        cls_pos_prob=0.5,
        train_cls_head=True)
    process = build_from_cfg(process, PIPELINES)

    num_frames = 60
    pair_video_infos = []

    filename = ['{:08d}.jpg'.format(i) for i in range(num_frames)]
    frame_ids = np.arange(num_frames)
    bboxes = np.ones((num_frames, 4))
    for video_id in range(2):
        bboxes_isvalid = np.ones(num_frames, dtype=bool)
        random_invalid_index = np.random.randint(0, num_frames, 4)
        bboxes_isvalid[random_invalid_index] = False
        visible = bboxes_isvalid.copy()
        random_invalid_index = np.random.randint(0, num_frames, 4)
        visible[random_invalid_index] = False
        video_info = dict(
            bboxes=bboxes,
            bboxes_isvalid=bboxes_isvalid,
            visible=visible,
            filename=filename,
            frame_ids=frame_ids,
            video_id=video_id)
        pair_video_infos.append(video_info)

    outs = process(pair_video_infos)
    if outs is not None:
        for out in outs:
            assert 0 <= out['img_info']['frame_id'] < num_frames
            assert 'labels' in out['ann_info']
            assert (out['ann_info']['bboxes'] == np.ones((1, 4))).all()


def test_pair_sampling():
    process = dict(
        type='PairSampling',
        frame_range=5,
        pos_prob=0.8,
        filter_template_img=False)
    process = build_from_cfg(process, PIPELINES)

    num_frames = 60
    pair_video_infos = []

    filename = ['{:08d}.jpg'.format(i) for i in range(num_frames)]
    frame_ids = np.arange(num_frames)
    bboxes = np.ones((num_frames, 4))
    for video_id in range(2):
        bboxes_isvalid = np.ones(num_frames, dtype=bool)
        visible = bboxes_isvalid.copy()
        video_info = dict(
            bboxes=bboxes,
            bboxes_isvalid=bboxes_isvalid,
            visible=visible,
            filename=filename,
            frame_ids=frame_ids,
            video_id=video_id)
        pair_video_infos.append(video_info)

    outs = process(pair_video_infos)
    if outs is not None:
        for out in outs:
            assert 0 <= out['img_info']['frame_id'] < num_frames
            assert 'is_positive_pairs' in out
            assert (out['ann_info']['bboxes'] == np.ones((1, 4))).all()


def test_match_instances():
    process = dict(type='MatchInstances', skip_nomatch=True)
    process = build_from_cfg(process, PIPELINES)

    results = [
        dict(gt_instance_ids=np.array([0, 1, 2, 3, 4])),
        dict(gt_instance_ids=np.array([2, 3, 4, 6]))
    ]
    outs = process(results)
    assert (outs[0]['gt_match_indices'] == np.array([-1, -1, 0, 1, 2])).all()
    assert (outs[1]['gt_match_indices'] == np.array([2, 3, 4, -1])).all()

    results = [
        dict(gt_instance_ids=np.array([0, 1, 2])),
        dict(gt_instance_ids=np.array([3, 4, 6, 7]))
    ]
    outs = process(results)
    assert outs is None

    process.skip_nomatch = False
    outs = process(results)
    assert (outs[0]['gt_match_indices'] == -1).all()
    assert (outs[1]['gt_match_indices'] == -1).all()
