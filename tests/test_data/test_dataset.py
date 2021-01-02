import logging
import os.path as osp
import tempfile
from collections import defaultdict
from unittest.mock import MagicMock, patch

import mmcv
import numpy as np
import pytest
import torch
import torch.nn as nn
from mmcv.runner import EpochBasedRunner
from torch.utils.data import DataLoader

from mmtrack.core.evaluation import DistEvalHook, EvalHook
from mmtrack.datasets import DATASETS

PREFIX = osp.join(osp.dirname(__file__), '../assets')
# This is a demo annotation file for CocoVideoDataset
# 1 videos, 2 categories ('car', 'person')
# 8 images, 2 instances -> [4, 3] objects
# 1 ignore, 2 crowd
DEMO_ANN_FILE = f'{PREFIX}/demo_cocovid_data/ann.json'
MOT_ANN_PATH = f'{PREFIX}/demo_mot17_data/'
LASOT_ANN_PATH = f'{PREFIX}/demo_sot_data/lasot'


def _create_gt_results(dataset):
    from mmdet.core import bbox2result

    from mmtrack.core import track2result
    results = defaultdict(list)
    for img_info in dataset.data_infos:
        ann = dataset.get_ann_info(img_info)
        scores = np.ones((ann['bboxes'].shape[0], 1), dtype=np.float)
        bboxes = np.concatenate((ann['bboxes'], scores), axis=1)
        bbox_results = bbox2result(bboxes, ann['labels'], len(dataset.CLASSES))
        track_results = track2result(bboxes, ann['labels'],
                                     ann['instance_ids'].astype(np.int),
                                     len(dataset.CLASSES))
        results['bbox_results'].append(bbox_results)
        results['track_results'].append(track_results)
    return results


@pytest.mark.parametrize('dataset', ['MOTChallengeDataset'])
def test_load_detections(dataset):
    dataset_class = DATASETS.get(dataset)
    dataset = dataset_class(
        ann_file=DEMO_ANN_FILE,
        classes=('car', 'person'),
        pipeline=[],
        test_mode=True)

    tmp_dir = tempfile.TemporaryDirectory()
    det_file = osp.join(tmp_dir.name, 'det.pkl')
    outputs = _create_gt_results(dataset)

    mmcv.dump(outputs['bbox_results'], det_file)
    detections = dataset.load_detections(det_file)
    assert isinstance(detections, list)
    assert len(detections) == 8

    mmcv.dump(outputs, det_file)
    detections = dataset.load_detections(det_file)
    assert isinstance(detections, list)
    assert len(detections) == 8
    dataset.detections = detections
    i = np.random.randint(0, len(dataset.data_infos))
    results = dataset.prepare_results(dataset.data_infos[i])
    assert 'detections' in results
    for a, b in zip(results['detections'], outputs['bbox_results'][i]):
        assert (a == b).all()

    out = dict()
    for i in range(len(dataset.data_infos)):
        out[dataset.data_infos[i]['file_name']] = outputs['bbox_results'][i]
    mmcv.dump(out, det_file)
    detections = dataset.load_detections(det_file)
    assert isinstance(detections, dict)
    assert len(detections) == 8
    dataset.detections = detections
    i = np.random.randint(0, len(dataset.data_infos))
    results = dataset.prepare_results(dataset.data_infos[i])
    assert 'detections' in results
    for a, b in zip(results['detections'], outputs['bbox_results'][i]):
        assert (a == b).all()

    tmp_dir.cleanup()


@pytest.mark.parametrize('dataset',
                         ['CocoVideoDataset', 'MOTChallengeDataset'])
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


@pytest.mark.parametrize('dataset', ['LaSOTDataset'])
def test_lasot_dataset_parse_ann_info(dataset):
    dataset_class = DATASETS.get(dataset)

    dataset = dataset_class(
        ann_file=osp.join(LASOT_ANN_PATH, 'lasot_test_dummy.json'),
        pipeline=[])

    # image 5 has 1 objects
    img_id = 5
    img_info = dataset.coco.load_imgs([img_id])[0]
    ann_ids = dataset.coco.get_ann_ids([img_id])
    ann_info = dataset.coco.loadAnns(ann_ids)
    ann = dataset._parse_ann_info(img_info, ann_info)
    assert ann['bboxes'].shape == (4, )
    assert ann['labels'] == 0


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
    dataset = dataset_class(
        ann_file=DEMO_ANN_FILE, classes=classes, pipeline=[])
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


def test_mot17_bbox_evaluation():
    classes = ('car', 'person')
    dataset_class = DATASETS.get('MOTChallengeDataset')
    dataset = dataset_class(
        ann_file=DEMO_ANN_FILE, classes=classes, pipeline=[])
    results = _create_gt_results(dataset)

    eval_results = dataset.evaluate(results, metric='bbox')
    assert eval_results['mAP'] == 1.0
    eval_results = dataset.evaluate(results['bbox_results'], metric='bbox')
    assert eval_results['mAP'] == 1.0


@patch('mmtrack.datasets.MOTChallengeDataset.load_annotations', MagicMock)
@patch('mmtrack.datasets.MOTChallengeDataset._filter_imgs', MagicMock)
@pytest.mark.parametrize('dataset', ['MOTChallengeDataset'])
def test_mot17_track_evaluation(dataset):
    tmp_dir = tempfile.TemporaryDirectory()
    videos = ['TUD-Campus', 'TUD-Stadtmitte']

    dataset_class = DATASETS.get(dataset)
    dataset_class.cat_ids = MagicMock()
    dataset_class.coco = MagicMock()

    dataset = dataset_class(
        ann_file=MagicMock(), visibility_thr=-1, pipeline=[])
    dataset.img_prefix = MOT_ANN_PATH
    dataset.vid_ids = [1, 2]
    vid_infos = [dict(name=_) for _ in videos]
    dataset.coco.load_vids = MagicMock(return_value=vid_infos)
    dataset.data_infos = []

    def _load_results(videos):
        track_results, data_infos = [], []
        for video in videos:
            dets = mmcv.list_from_file(
                osp.join(MOT_ANN_PATH, 'results', f'{video}.txt'))
            track_result = defaultdict(list)
            for det in dets:
                det = det.strip().split(',')
                frame_id, ins_id = map(int, det[:2])
                bbox = list(map(float, det[2:7]))
                track = [
                    ins_id, bbox[0], bbox[1], bbox[0] + bbox[2],
                    bbox[1] + bbox[3], bbox[4]
                ]
                track_result[frame_id].append(track)
            max_frame = max(track_result.keys())
            for i in range(1, max_frame + 1):
                track_results.append(
                    [np.array(track_result[i], dtype=np.float32)])
                data_infos.append(dict(frame_id=i - 1))
        return track_results, data_infos

    track_results, data_infos = _load_results(videos)
    dataset.data_infos = data_infos

    eval_results = dataset.evaluate(
        dict(track_results=track_results),
        metric='track',
        logger=None,
        resfile_path=None,
        track_iou_thr=0.5)
    assert eval_results['IDF1'] == 0.624
    assert eval_results['IDP'] == 0.799
    assert eval_results['MOTA'] == 0.555
    assert eval_results['IDs'] == 14

    tmp_dir.cleanup()


def test_lasot_evaluation():
    dataset_class = DATASETS.get('LaSOTDataset')
    dataset = dataset_class(
        ann_file=osp.join(LASOT_ANN_PATH, 'lasot_test_dummy.json'),
        pipeline=[])

    results = []
    for video_name in ['airplane-1', 'airplane-2']:
        results.extend(
            mmcv.list_from_file(
                osp.join(LASOT_ANN_PATH, video_name, 'track_results.txt')))
    track_results = []
    for result in results:
        x1, y1, x2, y2 = result.split(',')
        track_results.append(np.array([int(x1), int(y1), int(x2), int(y2)]))

    track_results = dict(bbox=track_results)
    eval_results = dataset.evaluate(track_results, metric=['track'])
    assert eval_results['success'] == 67.524
    assert eval_results['norm_precision'] == 70.0
    assert eval_results['precision'] == 50.0


@patch('mmtrack.apis.single_gpu_test', MagicMock)
@patch('mmtrack.apis.multi_gpu_test', MagicMock)
@pytest.mark.parametrize('EvalHookParam', (EvalHook, DistEvalHook))
def test_evaluation_hook(EvalHookParam):
    # create dummy data
    dataloader = DataLoader(torch.ones((5, 2)))
    dataloader.dataset.load_as_video = True

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
