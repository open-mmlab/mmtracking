import os.path as osp

import pytest
from mmcls.datasets import DATASETS

PREFIX = osp.join(osp.dirname(__file__), '../assets')
# This is a demo annotation file for ReIDDataset
REID_ANN_FILE = f'{PREFIX}/demo_reid_data/mot17_reid/ann.txt'


@pytest.mark.parametrize('dataset', ['ReIDDataset'])
def test_reid_dataset_parse_ann_info(dataset):
    dataset_class = DATASETS.get(dataset)

    dataset = dataset_class(
        data_prefix='reid', ann_file=REID_ANN_FILE, pipeline=[])
    data_infos = dataset.load_annotations()
    assert len(data_infos) == 704
