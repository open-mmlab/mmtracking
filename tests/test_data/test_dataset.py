import logging
import os.path as osp
import tempfile
from collections import defaultdict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from mmcv.runner import EpochBasedRunner
from torch.utils.data import DataLoader

from mmtrack.core.evaluation import DistEvalHook, EvalHook
from mmtrack.datasets import DATASETS, CocoVideoDataset

PREFIX = osp.join(osp.dirname(__file__), '../assets')
# This is a demo annotation file for CocoVideoDataset
# 1 videos, 2 categories ('car', 'person')
# 8 images, 2 instances, 13 objects, 1 ignore objects
COCO_VIDEO_ANN_FILE = f'{PREFIX}/demo_video_annotations_cocoformat.json'
MOT17_ANN_FILE = f'{PREFIX}/demo_mot17_cocoformat.json'


def _create_gt_results(dataset):
    from mmdet.core import bbox2result
    from mmtrack.core import track2result
    results = defaultdict(list)
    for img_info in dataset.data_infos:
        ann = dataset.get_ann_info(img_info)
        scores = np.ones((ann['bboxes'].shape[0], 1), dtype=np.float)
        bboxes = np.concatenate((ann['bboxes'], scores), axis=1)
        bbox_result = bbox2result(bboxes, ann['labels'], len(dataset.CLASSES))
        track_result = track2result(bboxes, ann['labels'],
                                    ann['instance_ids'].astype(np.int),
                                    len(dataset.CLASSES))
        results['bbox_result'].append(bbox_result)
        results['track_result'].append(track_result)
    return results


def test_mot17_dataset():
    dataset_class = DATASETS.get('MOT17Dataset')

    dataset = dataset_class(
        ann_file=MOT17_ANN_FILE, visibility_thr=-1, pipeline=[])
    # image 3 has 1 unvisualable object and 1 ignore object
    img_info = dataset.coco.load_imgs([3])[0]
    ann_ids = dataset.coco.get_ann_ids([3])
    ann_info = dataset.coco.loadAnns(ann_ids)
    ann = dataset._parse_ann_info(img_info, ann_info)
    assert ann['bboxes'].shape[0] == 1
    assert ann['bboxes'].shape[1] == 4
    assert ann['bboxes_ignore'].shape[0] == 1
    assert ann['public_bboxes'].shape[0] == 1
    assert ann['public_bboxes'].shape[1] == 5

    dataset.visibility_thr = 0.25
    ann = dataset._parse_ann_info(img_info, ann_info)
    assert ann['bboxes'].shape[0] == 0
    assert ann['bboxes_ignore'].shape[0] == 1
    assert ann['public_bboxes'].shape[0] == 1


@pytest.mark.parametrize('dataset', ['CocoVideoDataset', 'BDDVideoDataset'])
def test_video_data_sampling(dataset):
    dataset_class = DATASETS.get(dataset)

    # key image sampling
    for interval in [4, 2, 1]:
        dataset = dataset_class(
            ann_file=COCO_VIDEO_ANN_FILE,
            load_as_video=True,
            key_img_sampler=dict(interval=interval),
            ref_img_sampler=dict(
                num_ref_imgs=1,
                frame_range=3,
                filter_key_frame=True,
                method='uniform'),
            pipeline=[])
        assert len(dataset.data_infos) == 8 // interval

    # ref image sampling
    data = dataset.data_infos[3]
    sampler = dict(num_ref_imgs=1, frame_range=3, method='uniform')
    ref_data = dataset.ref_img_sampling(data, **sampler)[0]
    assert abs(ref_data['frame_id'] -
               data['frame_id']) <= sampler['frame_range']
    sampler = dict(num_ref_imgs=2, frame_range=3, method='bilateral_uniform')
    ref_data = dataset.ref_img_sampling(data, **sampler)
    assert len(ref_data) == 2
    assert ref_data[0]['frame_id'] < data['frame_id']
    assert ref_data[1]['frame_id'] > data['frame_id']
    assert data['frame_id'] - ref_data[0]['frame_id'] <= sampler['frame_range']
    assert ref_data[1]['frame_id'] - data['frame_id'] <= sampler['frame_range']


def test_dataset_evaluation():
    classes = ('car', 'person')
    dataset = CocoVideoDataset(
        ann_file=COCO_VIDEO_ANN_FILE, classes=classes, pipeline=[])
    results = _create_gt_results(dataset)
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
    dataset = CocoVideoDataset(
        ann_file=COCO_VIDEO_ANN_FILE, classes=classes, pipeline=[])
    results = _create_gt_results(dataset)
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


@patch('mmtrack.apis.single_gpu_test', MagicMock)
@patch('mmtrack.apis.multi_gpu_test', MagicMock)
@pytest.mark.parametrize('EvalHookParam', (EvalHook, DistEvalHook))
def test_evaluation_hook(EvalHookParam):
    # create dummy data
    dataloader = DataLoader(torch.ones((5, 2)))

    # 0.1. dataloader is not a DataLoader object
    with pytest.raises(TypeError):
        EvalHookParam(dataloader=MagicMock(), interval=-1)

    # 0.2. negative interval
    with pytest.raises(ValueError):
        EvalHookParam(dataloader, interval=-1)

    # 1. start=None, interval=1: perform evaluation after each epoch.
    runner = _build_demo_runner()
    evalhook = EvalHookParam(dataloader, interval=1)
    evalhook.evaluate = MagicMock()
    runner.register_hook(evalhook)
    runner.run([dataloader], [('train', 1)], 2)
    assert evalhook.evaluate.call_count == 2  # after epoch 1 & 2

    # 2. start=1, interval=1: perform evaluation after each epoch.
    runner = _build_demo_runner()

    evalhook = EvalHookParam(dataloader, start=1, interval=1)
    evalhook.evaluate = MagicMock()
    runner.register_hook(evalhook)
    runner.run([dataloader], [('train', 1)], 2)
    assert evalhook.evaluate.call_count == 2  # after epoch 1 & 2

    # 3. start=None, interval=2: perform evaluation after epoch 2, 4, 6, etc
    runner = _build_demo_runner()
    evalhook = EvalHookParam(dataloader, interval=2)
    evalhook.evaluate = MagicMock()
    runner.register_hook(evalhook)
    runner.run([dataloader], [('train', 1)], 2)
    assert evalhook.evaluate.call_count == 1  # after epoch 2

    # 4. start=1, interval=2: perform evaluation after epoch 1, 3, 5, etc
    runner = _build_demo_runner()
    evalhook = EvalHookParam(dataloader, start=1, interval=2)
    evalhook.evaluate = MagicMock()
    runner.register_hook(evalhook)
    runner.run([dataloader], [('train', 1)], 3)
    assert evalhook.evaluate.call_count == 2  # after epoch 1 & 3

    # 5. start=0/negative, interval=1: perform evaluation after each epoch and
    #    before epoch 1.
    runner = _build_demo_runner()
    evalhook = EvalHookParam(dataloader, start=0)
    evalhook.evaluate = MagicMock()
    runner.register_hook(evalhook)
    runner.run([dataloader], [('train', 1)], 2)
    assert evalhook.evaluate.call_count == 3  # before epoch1 and after e1 & e2

    runner = _build_demo_runner()
    with pytest.warns(UserWarning):
        evalhook = EvalHookParam(dataloader, start=-2)
    evalhook.evaluate = MagicMock()
    runner.register_hook(evalhook)
    runner.run([dataloader], [('train', 1)], 2)
    assert evalhook.evaluate.call_count == 3  # before epoch1 and after e1 & e2

    # 6. resuming from epoch i, start = x (x<=i), interval =1: perform
    #    evaluation after each epoch and before the first epoch.
    runner = _build_demo_runner()
    evalhook = EvalHookParam(dataloader, start=1)
    evalhook.evaluate = MagicMock()
    runner.register_hook(evalhook)
    runner._epoch = 2
    runner.run([dataloader], [('train', 1)], 3)
    assert evalhook.evaluate.call_count == 2  # before & after epoch 3

    # 7. resuming from epoch i, start = i+1/None, interval =1: perform
    #    evaluation after each epoch.
    runner = _build_demo_runner()
    evalhook = EvalHookParam(dataloader, start=2)
    evalhook.evaluate = MagicMock()
    runner.register_hook(evalhook)
    runner._epoch = 1
    runner.run([dataloader], [('train', 1)], 3)
    assert evalhook.evaluate.call_count == 2  # after epoch 2 & 3


def _build_demo_runner():

    class Model(nn.Module):

        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 1)

        def forward(self, x):
            return self.linear(x)

        def train_step(self, x, optimizer, **kwargs):
            return dict(loss=self(x))

        def val_step(self, x, optimizer, **kwargs):
            return dict(loss=self(x))

    model = Model()
    tmp_dir = tempfile.mkdtemp()

    runner = EpochBasedRunner(
        model=model, work_dir=tmp_dir, logger=logging.getLogger())
    return runner
