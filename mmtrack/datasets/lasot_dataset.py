# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmdet.datasets import DATASETS

from .sot_test_dataset import SOTTestDataset


@DATASETS.register_module()
class LaSOTDataset(SOTTestDataset):
    """LaSOT dataset for the testing of single object tracking.

    The dataset doesn't support training mode.
    """

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox annotations.

        Args:
            img_info (dict): image information.
            ann_info (list[dict]): Annotation information of an image. Each
                image only has one bbox annotation.

        Returns:
            dict: A dict containing the following keys: bboxes, labels,
            ignore. labels are not useful in SOT.
        """
        gt_bboxes = np.array(ann_info[0]['bbox'], dtype=np.float32)
        # convert [x1, y1, w, h] to [x1, y1, x2, y2]
        gt_bboxes[2] += gt_bboxes[0]
        gt_bboxes[3] += gt_bboxes[1]
        gt_labels = np.array(self.cat2label[ann_info[0]['category_id']])
        ignore = ann_info[0]['full_occlusion'] or ann_info[0]['out_of_view']
        ann = dict(bboxes=gt_bboxes, labels=gt_labels, ignore=ignore)
        return ann
