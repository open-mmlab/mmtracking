import numpy as np
from mmdet.datasets import DATASETS

from .sot_test_dataset import SOTTestDataset


@DATASETS.register_module()
class VOTDataset(SOTTestDataset):
    """VOT dataset for the testing of single object tracking.

    The dataset doesn't support training mode.

    Note: The vot datasets, such as VOT2020, using the mask annotation is not
    supported now.
    """
    CLASSES = (0, )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox annotations.

        Args:
            img_info (dict): image information.
            ann_info (list[dict]): Annotation information of an image. Each
                image only has one bbox annotation.
        Returns:
            dict: A dict containing the following keys: bboxes, labels.
            labels are not useful in SOT.
        """
        gt_bboxes = np.array(ann_info[0]['bbox'], dtype=np.float32)
        gt_labels = np.array(self.cat2label[ann_info[0]['category_id']])
        ann = dict(bboxes=gt_bboxes, labels=gt_labels)
        return ann

    def evaluate(self, results, metric=..., logger=None):
        pass
