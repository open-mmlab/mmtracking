# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmcv.utils import build_from_cfg

from mmtrack.datasets import PIPELINES


def test_match_instances():
    process = dict(type='MatchInstances', skip_nomatch=True)
    process = build_from_cfg(process, PIPELINES)

    results = [
        dict(gt_instance_ids=np.array([0, 1, 2, 3, 4])),
        dict(gt_instance_ids=np.array([2, 3, 4, 6]))
    ]
    outs = process(results)
    assert (outs[0]['gt_match_indices'] == np.array([-1, -1, 0, 1, 2])).all()
    assert (outs[1]['gt_match_indices'] == np.array([2, 3, 4, -1])).all()

    results = [
        dict(gt_instance_ids=np.array([0, 1, 2])),
        dict(gt_instance_ids=np.array([3, 4, 6, 7]))
    ]
    outs = process(results)
    assert outs is None

    process.skip_nomatch = False
    outs = process(results)
    assert (outs[0]['gt_match_indices'] == -1).all()
    assert (outs[1]['gt_match_indices'] == -1).all()
