# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import tempfile

import mmcv
import numpy as np
import pytest

from mmtrack.datasets import DATASETS as DATASETS

PREFIX = osp.join(osp.dirname(__file__), '../../data')
SOT_DATA_PREFIX = f'{PREFIX}/demo_sot_data'
DATASET_INFOS = dict(
    GOT10kDataset=dict(
        ann_file=osp.join(
            SOT_DATA_PREFIX,
            'trackingnet/annotations/trackingnet_train_infos.txt'),
        img_prefix=osp.join(SOT_DATA_PREFIX, 'trackingnet')),
    VOTDataset=dict(
        dataset_type='vot2018',
        ann_file=osp.join(
            SOT_DATA_PREFIX,
            'trackingnet/annotations/trackingnet_train_infos.txt'),
        img_prefix=osp.join(SOT_DATA_PREFIX, 'trackingnet')),
    OTB100Dataset=dict(
        ann_file=osp.join(
            SOT_DATA_PREFIX,
            'trackingnet/annotations/trackingnet_train_infos.txt'),
        img_prefix=osp.join(SOT_DATA_PREFIX, 'trackingnet')),
    UAV123Dataset=dict(
        ann_file=osp.join(
            SOT_DATA_PREFIX,
            'trackingnet/annotations/trackingnet_train_infos.txt'),
        img_prefix=osp.join(SOT_DATA_PREFIX, 'trackingnet')),
    LaSOTDataset=dict(
        ann_file=osp.join(
            SOT_DATA_PREFIX,
            'trackingnet/annotations/trackingnet_train_infos.txt'),
        img_prefix=osp.join(SOT_DATA_PREFIX, 'trackingnet')),
    TrackingNetDataset=dict(
        chunks_list=[0],
        ann_file=osp.join(
            SOT_DATA_PREFIX,
            'trackingnet/annotations/trackingnet_train_infos.txt'),
        img_prefix=osp.join(SOT_DATA_PREFIX, 'trackingnet')),
    SOTCocoDataset=dict(
        ann_file=osp.join(PREFIX, 'demo_cocovid_data', 'ann.json'),
        img_prefix=osp.join(PREFIX, 'demo_cocovid_data')),
    SOTImageNetVIDDataset=dict(
        ann_file=osp.join(PREFIX, 'demo_cocovid_data', 'ann.json'),
        img_prefix=osp.join(PREFIX, 'demo_cocovid_data')))


@pytest.mark.parametrize('dataset', [
    'GOT10kDataset', 'VOTDataset', 'OTB100Dataset', 'UAV123Dataset',
    'LaSOTDataset', 'TrackingNetDataset', 'SOTImageNetVIDDataset',
    'SOTCocoDataset'
])
def test_load_data_infos(dataset):
    dataset_class = DATASETS.get(dataset)

    dataset_class(
        **DATASET_INFOS[dataset], pipeline=[], split='train', test_mode=False)


@pytest.mark.parametrize('dataset', [
    'GOT10kDataset', 'VOTDataset', 'LaSOTDataset', 'TrackingNetDataset',
    'SOTImageNetVIDDataset', 'SOTCocoDataset'
])
def test_get_bboxes_from_video(dataset):
    dataset_class = DATASETS.get(dataset)

    dataset_object = dataset_class(
        **DATASET_INFOS[dataset], pipeline=[], split='train', test_mode=False)

    bboxes = dataset_object.get_bboxes_from_video(0)
    assert bboxes.shape[0] == dataset_object.num_frames_per_video[0]
    assert bboxes.shape[1] == 4


@pytest.mark.parametrize('dataset', [
    'GOT10kDataset', 'VOTDataset', 'LaSOTDataset', 'TrackingNetDataset',
    'SOTImageNetVIDDataset', 'SOTCocoDataset'
])
def test_get_visibility_from_video(dataset):
    dataset_class = DATASETS.get(dataset)

    dataset_object = dataset_class(
        **DATASET_INFOS[dataset], pipeline=[], split='train', test_mode=False)
    visibility = dataset_object.get_visibility_from_video(0)
    assert len(visibility['visible']) == dataset_object.num_frames_per_video[0]


@pytest.mark.parametrize('dataset', [
    'GOT10kDataset', 'TrackingNetDataset', 'SOTImageNetVIDDataset',
    'SOTCocoDataset', 'VOTDataset', 'LaSOTDataset'
])
def test_get_ann_infos_from_video(dataset):
    dataset_class = DATASETS.get(dataset)

    dataset_object = dataset_class(
        **DATASET_INFOS[dataset], pipeline=[], split='train', test_mode=False)
    dataset_object.get_ann_infos_from_video(0)


@pytest.mark.parametrize('dataset', [
    'GOT10kDataset', 'TrackingNetDataset', 'SOTImageNetVIDDataset',
    'SOTCocoDataset', 'VOTDataset', 'LaSOTDataset'
])
def test_get_img_infos_from_video(dataset):
    dataset_class = DATASETS.get(dataset)

    dataset_object = dataset_class(
        **DATASET_INFOS[dataset], pipeline=[], split='train', test_mode=False)
    dataset_object.get_img_infos_from_video(0)


@pytest.mark.parametrize(
    'dataset',
    ['GOT10kDataset', 'VOTDataset', 'LaSOTDataset', 'TrackingNetDataset'])
def test_prepare_test_data(dataset):
    dataset_class = DATASETS.get(dataset)

    dataset_object = dataset_class(
        **DATASET_INFOS[dataset], pipeline=[], split='train', test_mode=True)
    dataset_object.prepare_test_data(0, 1)


@pytest.mark.parametrize('dataset', [
    'GOT10kDataset', 'TrackingNetDataset', 'SOTImageNetVIDDataset',
    'SOTCocoDataset', 'LaSOTDataset'
])
def test_prepare_train_data(dataset):
    dataset_class = DATASETS.get(dataset)

    dataset_object = dataset_class(
        **DATASET_INFOS[dataset], pipeline=[], split='train', test_mode=False)
    dataset_object.prepare_train_data(0)


@pytest.mark.parametrize('dataset', ['GOT10kDataset', 'TrackingNetDataset'])
def test_format_results(dataset):
    dataset_class = DATASETS.get(dataset)

    dataset_object = dataset_class(
        **DATASET_INFOS[dataset], pipeline=[], split='train', test_mode=True)

    results = []
    for video_name in ['video-1', 'video-2']:
        results.extend(
            mmcv.list_from_file(
                osp.join(SOT_DATA_PREFIX, 'trackingnet', 'TRAIN_0', video_name,
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
    if osp.isfile(f'{tmp_dir.name}.zip'):
        os.remove(f'{tmp_dir.name}.zip')


def test_sot_ope_evaluation():
    dataset_class = DATASETS.get('UAV123Dataset')
    dataset_object = dataset_class(
        **DATASET_INFOS['UAV123Dataset'],
        pipeline=[],
        split='test',
        test_mode=True)

    dataset_object.num_frames_per_video = [25, 25]
    results = []
    data_infos = []
    data_root = osp.join(SOT_DATA_PREFIX, 'trackingnet', 'TRAIN_0')
    for video_name in ['video-1', 'video-2']:
        bboxes = np.loadtxt(
            osp.join(data_root, video_name, 'track_results.txt'),
            delimiter=',')
        scores = np.zeros((len(bboxes), 1))
        bboxes = np.concatenate((bboxes, scores), axis=-1)
        results.extend(bboxes)
        data_infos.append(
            dict(
                video_path=osp.join(data_root, video_name),
                ann_path=osp.join(data_root, video_name, 'gt_for_eval.txt'),
                start_frame_id=1,
                end_frame_id=25,
                framename_template='%06d.jpg'))

    dataset_object.data_infos = data_infos
    track_results = dict(track_bboxes=results)
    eval_results = dataset_object.evaluate(track_results, metric=['track'])
    assert eval_results['success'] == 67.524
    assert eval_results['norm_precision'] == 70.0
    assert eval_results['precision'] == 50.0


def test_sot_vot_evaluation():
    dataset_class = DATASETS.get('VOTDataset')
    dataset_object = dataset_class(
        **DATASET_INFOS['VOTDataset'],
        pipeline=[],
        split='test',
        test_mode=True)

    dataset_object.num_frames_per_video = [25, 25]
    data_infos = []
    results = []
    vot_root = osp.join(SOT_DATA_PREFIX, 'trackingnet', 'TRAIN_0')
    for video_name in ['video-1', 'video-2']:
        results.extend(
            mmcv.list_from_file(
                osp.join(vot_root, video_name, 'vot2018_track_results.txt')))
        data_infos.append(
            dict(
                video_path=osp.join(vot_root, video_name),
                ann_path=osp.join(vot_root, video_name,
                                  'vot2018_gt_for_eval.txt'),
                start_frame_id=1,
                end_frame_id=25,
                framename_template='%08d.jpg'))
    dataset_object.data_infos = data_infos

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
    eval_results = dataset_object.evaluate(
        track_bboxes, interval=[1, 3], metric=['track'])
    assert abs(eval_results['eao'] - 0.6661) < 0.0001
    assert round(eval_results['accuracy'], 4) == 0.5826
    assert round(eval_results['robustness'], 4) == 6.0
