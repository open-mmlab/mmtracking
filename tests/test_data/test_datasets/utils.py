from collections import defaultdict

import numpy as np


def _create_coco_gt_results(dataset):
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
