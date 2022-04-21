# Copyright (c) OpenMMLab. All rights reserved.

import os.path as osp

from mmtrack.core.evaluation import eval_vis
from mmtrack.core.utils import YTVOS

PREFIX = osp.join(osp.dirname(__file__), '../../data')

DEMO_ANN_FILE = f'{PREFIX}/demo_vis_data/ann.json'
DEMO_RES_FILE = f'{PREFIX}/demo_vis_data/results.json'


def test_eval_vis():
    eval_results = eval_vis(DEMO_RES_FILE, DEMO_ANN_FILE, None)
    assert eval_results is not None
    assert len(eval_results) == 7

    ytvos = YTVOS(DEMO_ANN_FILE)
    assert isinstance(ytvos, YTVOS)

    ytvos.anns[1]['iscrowd'] = 1
    ids = ytvos.getAnnIds()
    assert ids is not None

    res = ytvos.loadAnns(1)
    assert res is not None
    res = ytvos.loadCats(1)
    assert res is not None
    res = ytvos.loadVids(1)
    assert res is not None
