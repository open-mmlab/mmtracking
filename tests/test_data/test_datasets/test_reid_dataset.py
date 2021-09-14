# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import pytest
import torch

from mmtrack.datasets import DATASETS as DATASETS

PREFIX = osp.join(osp.dirname(__file__), '../../data')
# This is a demo annotation file for ReIDDataset
REID_ANN_FILE = f'{PREFIX}/demo_reid_data/mot17_reid/ann.txt'


def _create_reid_gt_results(dataset):
    results = []
    dataset_infos = dataset.load_annotations()
    for dataset_info in dataset_infos:
        result = torch.full((128, ),
                            float(dataset_info['gt_label']),
                            dtype=torch.float32)
        results.append(result)
    return results


@pytest.mark.parametrize('dataset', ['ReIDDataset'])
def test_reid_dataset_parse_ann_info(dataset):
    dataset_class = DATASETS.get(dataset)

    dataset = dataset_class(
        data_prefix='reid', ann_file=REID_ANN_FILE, pipeline=[])
    data_infos = dataset.load_annotations()
    img_id = 0
    # image 0 has 21 objects
    assert len([
        data_info for data_info in data_infos
        if data_info['gt_label'] == img_id
    ]) == 21
    img_id = 11
    # image 11 doesn't have objects
    assert len([
        data_info for data_info in data_infos
        if data_info['gt_label'] == img_id
    ]) == 0


@pytest.mark.parametrize('dataset', ['ReIDDataset'])
def test_reid_dataset_prepare_data(dataset):
    dataset_class = DATASETS.get(dataset)

    num_ids = 8
    ins_per_id = 4
    dataset = dataset_class(
        data_prefix='reid',
        ann_file=REID_ANN_FILE,
        triplet_sampler=dict(num_ids=num_ids, ins_per_id=ins_per_id),
        pipeline=[],
        test_mode=False)
    assert len(dataset) == 704

    results = dataset.prepare_data(0)
    assert isinstance(results, list)
    assert len(results) == 32
    assert 'img_info' in results[0]
    assert 'gt_label' in results[0]
    assert results[0].keys() == results[1].keys()
    # triplet sampling
    for idx in range(len(results) - 1):
        if (idx + 1) % ins_per_id != 0:
            assert results[idx]['gt_label'] == results[idx + 1]['gt_label']


@pytest.mark.parametrize('dataset', ['ReIDDataset'])
def test_reid_evaluation(dataset):
    dataset_class = DATASETS.get(dataset)

    dataset = dataset_class(
        data_prefix='reid', ann_file=REID_ANN_FILE, pipeline=[])
    results = _create_reid_gt_results(dataset)
    eval_results = dataset.evaluate(results, metric=['mAP', 'CMC'])
    assert eval_results['mAP'] == 1
    assert eval_results['R1'] == 1
    assert eval_results['R5'] == 1
    assert eval_results['R10'] == 1
    assert eval_results['R20'] == 1
