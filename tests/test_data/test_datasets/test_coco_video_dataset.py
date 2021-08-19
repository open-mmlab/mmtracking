# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import pytest

from mmtrack.datasets import DATASETS as DATASETS
from .utils import _create_coco_gt_results

PREFIX = osp.join(osp.dirname(__file__), '../../data')
# This is a demo annotation file for CocoVideoDataset
# 1 videos, 2 categories ('car', 'person')
# 8 images, 2 instances -> [4, 3] objects
# 1 ignore, 2 crowd
DEMO_ANN_FILE = f'{PREFIX}/demo_cocovid_data/ann.json'


@pytest.mark.parametrize('dataset', ['CocoVideoDataset'])
def test_parse_ann_info(dataset):
    dataset_class = DATASETS.get(dataset)

    dataset = dataset_class(
        ann_file=DEMO_ANN_FILE, classes=('car', 'person'), pipeline=[])

    # image 1 doesn't have gt and detected objects
    img_id = 1
    img_info = dataset.coco.load_imgs([img_id])[0]
    ann_ids = dataset.coco.get_ann_ids([img_id])
    ann_info = dataset.coco.loadAnns(ann_ids)
    ann = dataset._parse_ann_info(img_info, ann_info)
    assert ann['bboxes'].shape == (0, 4)
    assert ann['bboxes_ignore'].shape == (3, 4)

    # image 5 has 2 objects
    img_id = 5
    img_info = dataset.coco.load_imgs([img_id])[0]
    ann_ids = dataset.coco.get_ann_ids([img_id])
    ann_info = dataset.coco.loadAnns(ann_ids)
    ann = dataset._parse_ann_info(img_info, ann_info)
    assert ann['bboxes'].shape == (2, 4)
    assert ann['bboxes_ignore'].shape == (0, 4)

    # image 8 doesn't have objects
    img_id = 8
    img_info = dataset.coco.load_imgs([img_id])[0]
    ann_ids = dataset.coco.get_ann_ids([img_id])
    ann_info = dataset.coco.loadAnns(ann_ids)
    ann = dataset._parse_ann_info(img_info, ann_info)
    assert ann['bboxes'].shape == (0, 4)
    assert ann['bboxes_ignore'].shape == (0, 4)


@pytest.mark.parametrize('dataset', ['CocoVideoDataset'])
def test_prepare_data(dataset):
    dataset_class = DATASETS.get(dataset)

    # train
    dataset = dataset_class(
        ann_file=DEMO_ANN_FILE,
        classes=['car', 'person'],
        ref_img_sampler=dict(
            num_ref_imgs=1,
            frame_range=1,
            filter_key_img=True,
            method='uniform'),
        pipeline=[],
        test_mode=False)
    assert len(dataset) == 7

    results = dataset.prepare_train_img(0)
    assert isinstance(results, list)
    assert len(results) == 2
    assert 'ann_info' in results[0]
    assert results[0].keys() == results[1].keys()

    dataset.ref_img_sampler = None
    results = dataset.prepare_train_img(0)
    assert isinstance(results, dict)
    assert 'ann_info' in results

    # test
    dataset = dataset_class(
        ann_file=DEMO_ANN_FILE,
        classes=['car', 'person'],
        ref_img_sampler=dict(
            num_ref_imgs=1,
            frame_range=1,
            filter_key_img=True,
            method='uniform'),
        pipeline=[],
        test_mode=True)
    assert len(dataset) == 8

    results = dataset.prepare_test_img(0)
    assert isinstance(results, list)
    assert len(results) == 2
    assert 'ann_info' not in results[0]
    assert results[0].keys() == results[1].keys()

    dataset.ref_img_sampler = None
    results = dataset.prepare_test_img(0)
    assert isinstance(results, dict)
    assert 'ann_info' not in results


@pytest.mark.parametrize('dataset', ['CocoVideoDataset'])
def test_video_data_sampling(dataset):
    dataset_class = DATASETS.get(dataset)

    # key image sampling
    for interval in [4, 2, 1]:
        dataset = dataset_class(
            ann_file=DEMO_ANN_FILE,
            load_as_video=True,
            classes=['car', 'person'],
            key_img_sampler=dict(interval=interval),
            ref_img_sampler=dict(
                num_ref_imgs=1,
                frame_range=3,
                filter_key_frame=True,
                method='uniform'),
            pipeline=[],
            test_mode=True)
        assert len(dataset.data_infos) == 8 // interval

    # ref image sampling
    data = dataset.data_infos[3]
    sampler = dict(num_ref_imgs=1, frame_range=3, method='uniform')
    ref_data = dataset.ref_img_sampling(data, **sampler)[1]
    assert abs(ref_data['frame_id'] -
               data['frame_id']) <= sampler['frame_range']
    sampler = dict(num_ref_imgs=2, frame_range=3, method='bilateral_uniform')
    ref_data = dataset.ref_img_sampling(data, **sampler)
    assert len(ref_data) == 3
    ref_data = dataset.ref_img_sampling(data, **sampler, return_key_img=False)
    assert len(ref_data) == 2
    assert ref_data[0]['frame_id'] < data['frame_id']
    assert ref_data[1]['frame_id'] > data['frame_id']
    assert data['frame_id'] - ref_data[0]['frame_id'] <= sampler['frame_range']
    assert ref_data[1]['frame_id'] - data['frame_id'] <= sampler['frame_range']


def test_coco_video_evaluation():
    classes = ('car', 'person')
    dataset_class = DATASETS.get('CocoVideoDataset')
    dataset = dataset_class(
        ann_file=DEMO_ANN_FILE, classes=classes, pipeline=[])
    results = _create_coco_gt_results(dataset)
    eval_results = dataset.evaluate(results, metric=['bbox', 'track'])
    assert eval_results['bbox_mAP'] == 1.0
    assert eval_results['bbox_mAP_50'] == 1.0
    assert eval_results['bbox_mAP_75'] == 1.0
    assert 'bbox_mAP_copypaste' in eval_results
    assert eval_results['MOTA'] == 1.0
    assert eval_results['IDF1'] == 1.0
    assert eval_results['MT'] == 2
    assert 'track_OVERALL_copypaste' in eval_results
    assert 'track_AVERAGE_copypaste' in eval_results

    classes = ('car', )
    dataset = dataset_class(
        ann_file=DEMO_ANN_FILE, classes=classes, pipeline=[])
    results = _create_coco_gt_results(dataset)
    eval_results = dataset.evaluate(results, metric=['bbox', 'track'])
    assert eval_results['bbox_mAP'] == 1.0
    assert eval_results['bbox_mAP_50'] == 1.0
    assert eval_results['bbox_mAP_75'] == 1.0
    assert 'bbox_mAP_copypaste' in eval_results
    assert eval_results['MOTA'] == 1.0
    assert eval_results['IDF1'] == 1.0
    assert eval_results['MT'] == 1
    assert 'track_OVERALL_copypaste' in eval_results
    assert 'track_AVERAGE_copypaste' in eval_results
