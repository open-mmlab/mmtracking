# Copyright (c) OpenMMLab. All rights reserved.
from collections import defaultdict

import numpy as np


def _create_coco_gt_results(dataset):
    from mmtrack.core import outs2results

    results = defaultdict(list)
    for img_info in dataset.data_infos:
        ann = dataset.get_ann_info(img_info)
        scores = np.ones((ann['bboxes'].shape[0], 1), dtype=np.float)
        bboxes = np.concatenate((ann['bboxes'], scores), axis=1)
        det_outs_dict = dict(
            bboxes=bboxes,
            label=ann['labels'],
            num_classes=len(dataset.CLASSES))
        det_results_dict = outs2results(det_outs_dict)
        track_outs_dict = dict(
            bboxes=bboxes,
            labels=ann['labels'],
            ids=ann['instance_ids'].astype(np.int),
            num_classes=len(dataset.CLASSES))
        track_results_dict = outs2results(track_outs_dict)
        results['bbox_results'].append(det_results_dict['bboxes'])
        results['track_results'].append(track_results_dict['bboxes'])
    return results
