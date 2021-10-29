# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile

import mmcv
import numpy as np
import pytest

from mmtrack.datasets import DATASETS as DATASETS

PREFIX = osp.join(osp.dirname(__file__), '../../data')
LASOT_ANN_PATH = f'{PREFIX}/demo_sot_data/lasot'
VOT_ANN_PATH = f'{PREFIX}/demo_sot_data/vot'


@pytest.mark.parametrize('dataset',
                         ['SOTTestDataset', 'LaSOTDataset', 'VOTDataset'])
def test_parse_ann_info(dataset):
    dataset_class = DATASETS.get(dataset)

    ann_file = osp.join(
        LASOT_ANN_PATH,
        'lasot_test_dummy.json') if dataset != 'VOTDataset' else osp.join(
            VOT_ANN_PATH, 'vot_test_dummy.json')
    dataset_object = dataset_class(ann_file=ann_file, pipeline=[])

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
