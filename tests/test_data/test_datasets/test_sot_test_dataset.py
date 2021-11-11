# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile

import mmcv
import numpy as np
import pytest

from mmtrack.datasets import DATASETS as DATASETS

PREFIX = osp.join(osp.dirname(__file__), '../../data')
LASOT_ANN_PATH = f'{PREFIX}/demo_sot_data/lasot'


@pytest.mark.parametrize('dataset',
                         ['SOTTestDataset', 'LaSOTDataset', 'VOTDataset'])
def test_parse_ann_info(dataset):
    dataset_class = DATASETS.get(dataset)

    ann_file = osp.join(LASOT_ANN_PATH, 'lasot_test_dummy.json')
    dataset_object = dataset_class(ann_file=ann_file, pipeline=[])

    if dataset == 'VOTDataset':
        for _, img_ann in dataset_object.coco.anns.items():
            x, y, w, h = img_ann['bbox']
            img_ann['bbox'] = [x, y, x + w, y, x + w, y + h, x, y + h]

    # image 5 has 1 objects
    img_id = 5
    img_info = dataset_object.coco.load_imgs([img_id])[0]
    ann_ids = dataset_object.coco.get_ann_ids([img_id])
    ann_info = dataset_object.coco.loadAnns(ann_ids)
    ann = dataset_object._parse_ann_info(img_info, ann_info)
    assert ann['bboxes'].shape == (
        4, ) if dataset != 'VOTDataset' else ann['bboxes'].shape == (8, )
    assert ann['labels'] == 0


def test_sot_ope_evaluation():
    dataset_class = DATASETS.get('SOTTestDataset')
    dataset = dataset_class(
        ann_file=osp.join(LASOT_ANN_PATH, 'lasot_test_dummy.json'),
        pipeline=[])

    results = []
    for video_name in ['airplane-1', 'airplane-2']:
        results.extend(
            mmcv.list_from_file(
                osp.join(LASOT_ANN_PATH, video_name, 'track_results.txt')))
    track_bboxes = []
    for result in results:
        x1, y1, x2, y2 = result.split(',')
        track_bboxes.append(
            np.array([float(x1),
                      float(y1),
                      float(x2),
                      float(y2), 0.]))

    track_results = dict(track_bboxes=track_bboxes)
    eval_results = dataset.evaluate(track_results, metric=['track'])
    assert eval_results['success'] == 67.524
    assert eval_results['norm_precision'] == 70.0
    assert eval_results['precision'] == 50.0


def test_sot_vot_evaluation():
    dataset_class = DATASETS.get('VOTDataset')
    dataset = dataset_class(
        ann_file=osp.join(LASOT_ANN_PATH, 'lasot_test_dummy.json'),
        pipeline=[])

    for _, img_ann in dataset.coco.anns.items():
        x, y, w, h = img_ann['bbox']
        img_ann['bbox'] = [x, y, x + w, y, x + w, y + h, x, y + h]

    results = []
    for video_name in ['airplane-1', 'airplane-2']:
        results.extend(
            mmcv.list_from_file(
                osp.join(LASOT_ANN_PATH, video_name, 'vot_track_results.txt')))
    track_bboxes = []
    for result in results:
        result = result.split(',')
        if len(result) == 1:
            track_bboxes.append(np.array([float(result[0]), 0.]))
        else:
            track_bboxes.append(
                np.array([
                    float(result[0]),
                    float(result[1]),
                    float(result[2]),
                    float(result[3]), 0.
                ]))

    track_bboxes = dict(track_bboxes=track_bboxes)
    eval_results = dataset.evaluate(
        track_bboxes, interval=[1, 3], metric=['track'])
    assert abs(eval_results['eao'] - 0.6394) < 0.0001
    assert round(eval_results['accuracy'], 4) == 0.5431
    assert round(eval_results['robustness'], 4) == 6.0


@pytest.mark.parametrize('dataset', ['TrackingNetDataset', 'GOT10kDataset'])
def test_format_results(dataset):
    dataset_class = DATASETS.get(dataset)
    dataset = dataset_class(
        ann_file=osp.join(LASOT_ANN_PATH, 'lasot_test_dummy.json'),
        pipeline=[])

    results = []
    for video_name in ['airplane-1', 'airplane-2']:
        results.extend(
            mmcv.list_from_file(
                osp.join(LASOT_ANN_PATH, video_name, 'track_results.txt')))

    track_bboxes = []
    for result in results:
        x1, y1, x2, y2 = result.split(',')
        track_bboxes.append(
            np.array([float(x1),
                      float(y1),
                      float(x2),
                      float(y2), 0.]))

    track_results = dict(track_bboxes=track_bboxes)

    tmp_dir = tempfile.TemporaryDirectory()
    dataset.format_results(track_results, resfile_path=tmp_dir.name)
    if osp.isdir(tmp_dir.name):
        tmp_dir.cleanup()
