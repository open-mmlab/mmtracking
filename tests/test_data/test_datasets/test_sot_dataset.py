# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile

import mmcv
import numpy as np
import pytest

from mmtrack.datasets import DATASETS as DATASETS

PREFIX = osp.join(osp.dirname(__file__), '../../data')
SOT_DATA_PREFIX = f'{PREFIX}/demo_sot_data'
DATASET_INFOS = dict(
    GOT10kDataset=(osp.join(SOT_DATA_PREFIX, 'got10k'), ),
    VOTDataset=(2018, osp.join(SOT_DATA_PREFIX, 'vot2018')),
    OTB100Dataset=('tools/convert_datasets/otb100/otb100_infos.txt',
                   osp.join(SOT_DATA_PREFIX, 'otb100')),
    UAV123Dataset=('tools/convert_datasets/uav123/uav123_infos.txt',
                   osp.join(SOT_DATA_PREFIX, 'uav123')),
    LaSOTDataset=(osp.join(SOT_DATA_PREFIX, 'lasot_full', 'testing_set.txt'),
                  osp.join(SOT_DATA_PREFIX, 'lasot_full')))


@pytest.mark.parametrize('dataset', [
    'GOT10kDataset', 'VOTDataset', 'OTB100Dataset', 'UAV123Dataset',
    'LaSOTDataset'
])
def test_load_data_infos(dataset):
    dataset_class = DATASETS.get(dataset)

    dataset_class(
        *DATASET_INFOS[dataset], pipeline=[], split='train', test_mode=False)


@pytest.mark.parametrize('dataset',
                         ['GOT10kDataset', 'VOTDataset', 'LaSOTDataset'])
def test_get_bboxes_from_video(dataset):
    dataset_class = DATASETS.get(dataset)

    dataset_object = dataset_class(
        *DATASET_INFOS[dataset], pipeline=[], split='train', test_mode=False)

    bboxes = dataset_object.get_bboxes_from_video(0)
    if dataset == 'VOTDataset':
        assert len(bboxes.shape) == 2 and bboxes.shape[1] == 8
    else:
        assert len(bboxes.shape) == 2 and bboxes.shape[1] == 4


@pytest.mark.parametrize('dataset',
                         ['GOT10kDataset', 'VOTDataset', 'LaSOTDataset'])
def test_get_img_names_from_video(dataset):
    dataset_class = DATASETS.get(dataset)

    dataset_object = dataset_class(
        *DATASET_INFOS[dataset], pipeline=[], split='train', test_mode=False)
    image_names = dataset_object.get_img_names_from_video(0)
    assert len(image_names) == 2


@pytest.mark.parametrize('dataset',
                         ['GOT10kDataset', 'VOTDataset', 'LaSOTDataset'])
def test_get_visibility_from_video(dataset):
    dataset_class = DATASETS.get(dataset)

    dataset_object = dataset_class(
        *DATASET_INFOS[dataset], pipeline=[], split='train', test_mode=False)
    visibility = dataset_object.get_visibility_from_video(0)
    assert len(visibility['visible']) == 2


@pytest.mark.parametrize('dataset',
                         ['GOT10kDataset', 'VOTDataset', 'LaSOTDataset'])
def test_get_ann_infos_from_video(dataset):
    dataset_class = DATASETS.get(dataset)

    dataset_object = dataset_class(
        *DATASET_INFOS[dataset], pipeline=[], split='train', test_mode=False)
    dataset_object.get_ann_infos_from_video(0)


@pytest.mark.parametrize('dataset',
                         ['GOT10kDataset', 'VOTDataset', 'LaSOTDataset'])
def test_get_img_infos_from_video(dataset):
    dataset_class = DATASETS.get(dataset)

    dataset_object = dataset_class(
        *DATASET_INFOS[dataset], pipeline=[], split='train', test_mode=False)
    dataset_object.get_img_infos_from_video(0)


@pytest.mark.parametrize('dataset',
                         ['GOT10kDataset', 'VOTDataset', 'LaSOTDataset'])
def test_prepare_test_data(dataset):
    dataset_class = DATASETS.get(dataset)

    dataset_object = dataset_class(
        *DATASET_INFOS[dataset], pipeline=[], split='train', test_mode=True)
    dataset_object.prepare_test_data(0, 1)


@pytest.mark.parametrize('dataset', ['GOT10kDataset', 'LaSOTDataset'])
def test_prepare_train_data(dataset):
    dataset_class = DATASETS.get(dataset)

    dataset_object = dataset_class(
        *DATASET_INFOS[dataset], pipeline=[], split='train', test_mode=False)
    dataset_object.prepare_train_data(0)


@pytest.mark.parametrize('dataset', ['GOT10kDataset'])
def test_format_results(dataset):
    dataset_class = DATASETS.get(dataset)

    dataset_object = dataset_class(
        *DATASET_INFOS[dataset], pipeline=[], split='train', test_mode=True)

    results = []
    for video_name in ['airplane-1', 'airplane-2']:
        results.extend(
            mmcv.list_from_file(
                osp.join(SOT_DATA_PREFIX, 'lasot', video_name,
                         'track_results.txt')))

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
    dataset_object.format_results(track_results, resfile_path=tmp_dir.name)
    if osp.isdir(tmp_dir.name):
        tmp_dir.cleanup()
