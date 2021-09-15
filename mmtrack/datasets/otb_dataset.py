from mmtrack.datasets import DATASETS
from .sot_test_dataset import SOTTestDataset


@DATASETS.register_module()
class OTB100Dataset(SOTTestDataset):
    """OTB100 dataset for the testing of single object tracking.

    The dataset doesn't support training mode.
    """
    pass
