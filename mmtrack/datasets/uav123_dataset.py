# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.datasets import DATASETS

from .sot_test_dataset import SOTTestDataset


@DATASETS.register_module()
class UAV123Dataset(SOTTestDataset):
    """UAV123 dataset for the testing of single object tracking.

    The dataset doesn't support training mode.
    """
    pass
