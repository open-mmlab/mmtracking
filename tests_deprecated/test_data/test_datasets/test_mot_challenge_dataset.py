# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
from collections import defaultdict
from unittest.mock import MagicMock, patch

import mmcv
import numpy as np
import pytest

from mmtrack.datasets import DATASETS as DATASETS
from .utils import _create_coco_gt_results

PREFIX = osp.join(osp.dirname(__file__), '../../data')
# This is a demo annotation file for CocoVideoDataset
# 1 videos, 2 categories ('car', 'person')
# 8 images, 2 instances -> [4, 3] objects
# 1 ignore, 2 crowd
DEMO_ANN_FILE = f'{PREFIX}/demo_cocovid_data/ann.json'
MOT_ANN_PATH = f'{PREFIX}/demo_MOT15_data/train'


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
    outputs = _create_coco_gt_results(dataset)

    mmcv.dump(outputs['det_bboxes'], det_file)
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
    for a, b in zip(results['detections'], outputs['det_bboxes'][i]):
        assert (a == b).all()

    out = dict()
    for i in range(len(dataset.data_infos)):
        out[dataset.data_infos[i]['file_name']] = outputs['det_bboxes'][i]
    mmcv.dump(out, det_file)
    detections = dataset.load_detections(det_file)
    assert isinstance(detections, dict)
    assert len(detections) == 8
    dataset.detections = detections
    i = np.random.randint(0, len(dataset.data_infos))
    results = dataset.prepare_results(dataset.data_infos[i])
    assert 'detections' in results
    for a, b in zip(results['detections'], outputs['det_bboxes'][i]):
        assert (a == b).all()

    tmp_dir.cleanup()


@pytest.mark.parametrize('dataset', ['MOTChallengeDataset'])
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


def test_mot15_bbox_evaluation():
    classes = ('car', 'person')
    dataset_class = DATASETS.get('MOTChallengeDataset')
    dataset = dataset_class(
        ann_file=DEMO_ANN_FILE, classes=classes, pipeline=[])
    results = _create_coco_gt_results(dataset)

    eval_results = dataset.evaluate(results, metric='bbox')
    assert eval_results['mAP'] == 1.0
    eval_results = dataset.evaluate(results['det_bboxes'], metric='bbox')
    assert eval_results['mAP'] == 1.0


@patch('mmtrack.datasets.MOTChallengeDataset.load_annotations', MagicMock)
@patch('mmtrack.datasets.MOTChallengeDataset._filter_imgs', MagicMock)
@pytest.mark.parametrize('dataset', ['MOTChallengeDataset'])
def test_mot15_track_evaluation(dataset):
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
        track_bboxes, data_infos = [], []
        for video in videos:
            dets = mmcv.list_from_file(
                osp.join(MOT_ANN_PATH, 'results', f'{video}.txt'))
            track_bbox = defaultdict(list)
            for det in dets:
                det = det.strip().split(',')
                frame_id, ins_id = map(int, det[:2])
                bbox = list(map(float, det[2:7]))
                track = [
                    ins_id, bbox[0], bbox[1], bbox[0] + bbox[2],
                    bbox[1] + bbox[3], bbox[4]
                ]
                track_bbox[frame_id].append(track)
            max_frame = max(track_bbox.keys())
            for i in range(1, max_frame + 1):
                track_bboxes.append(
                    [np.array(track_bbox[i], dtype=np.float32)])
                data_infos.append(dict(frame_id=i - 1))
        return track_bboxes, data_infos

    track_bboxes, data_infos = _load_results(videos)
    dataset.data_infos = data_infos

    eval_results = dataset.evaluate(
        dict(track_bboxes=track_bboxes),
        metric='track',
        logger=None,
        resfile_path=None,
        track_iou_thr=0.5)
    assert eval_results['IDF1'] == 0.624
    assert eval_results['IDP'] == 0.799
    assert eval_results['MOTA'] == 0.555
    assert eval_results['IDs'] == 14
    assert eval_results['HOTA'] == 0.400

    tmp_dir.cleanup()
