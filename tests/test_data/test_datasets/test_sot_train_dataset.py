# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import pytest

from mmtrack.datasets import DATASETS as DATASETS

PREFIX = osp.join(osp.dirname(__file__), '../../data')
# This is a demo annotation file for CocoVideoDataset
# 1 videos, 2 categories ('car', 'person')
# 8 images, 2 instances -> [4, 3] objects
# 1 ignore, 2 crowd
DEMO_ANN_FILE = f'{PREFIX}/demo_cocovid_data/ann.json'


@pytest.mark.parametrize('dataset', ['SOTTrainDataset'])
def test_sot_train_dataset_parse_ann_info(dataset):
    dataset_class = DATASETS.get(dataset)

    dataset = dataset_class(ann_file=DEMO_ANN_FILE, pipeline=[])

    # image 5 has 2 objects, we only load the object with instance_id = 1
    img_id = 5
    instance_id = 1
    ann_ids = dataset.coco.get_ann_ids([img_id])
    ann_info = dataset.coco.loadAnns(ann_ids)
    ann = dataset._parse_ann_info(instance_id, ann_info)
    assert ann['bboxes'].shape == (1, 4)
    assert ann['labels'].shape == (1, ) and ann['labels'][0] == 0


@pytest.mark.parametrize('dataset', ['SOTTrainDataset'])
def test_sot_train_dataset_prepare_data(dataset):
    dataset_class = DATASETS.get(dataset)

    # train
    dataset = dataset_class(
        ann_file=DEMO_ANN_FILE,
        ref_img_sampler=dict(
            frame_range=100,
            pos_prob=0.8,
            filter_key_img=False,
            return_key_img=True),
        pipeline=[],
        test_mode=False)
    assert len(dataset) == 1

    results = dataset.prepare_train_img(0)
    assert isinstance(results, list)
    assert len(results) == 2
    assert 'ann_info' in results[0]
    assert results[0].keys() == results[1].keys()
