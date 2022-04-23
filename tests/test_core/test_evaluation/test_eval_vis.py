# Copyright (c) OpenMMLab. All rights reserved.

import os.path as osp

import mmcv

from mmtrack.core.evaluation import eval_vis
from mmtrack.core.utils import YTVIS

PREFIX = osp.join(osp.dirname(__file__), '../../data')

DEMO_ANN_FILE = f'{PREFIX}/demo_vis_data/ann.json'
DEMO_RES_FILE = f'{PREFIX}/demo_vis_data/results.json'


def test_eval_vis():
    json_results = mmcv.load(DEMO_RES_FILE)
    eval_results = eval_vis(json_results, DEMO_ANN_FILE, None)
    assert eval_results is not None
    assert len(eval_results) == 7

    ytvis = YTVIS(DEMO_ANN_FILE)
    assert isinstance(ytvis, YTVIS)

    ytvis.anns[1]['iscrowd'] = 1
    ids = ytvis.getAnnIds()
    assert ids is not None

    res = ytvis.loadAnns(1)
    assert res is not None
    res = ytvis.loadCats(1)
    assert res is not None
    res = ytvis.loadVids(1)
    assert res is not None
