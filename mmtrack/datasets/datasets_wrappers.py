import random

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.dataset_wrappers import ConcatDataset


@DATASETS.register_module()
class RandomConcatDataset(ConcatDataset):
    """A wrapper of concatenated dataset.

    Same as :obj:`torch.utils.data.dataset.ConcatDataset`, but
    concat the group flag for image aspect ratio.

    Args:
        datasets (list[:obj:`Dataset`]): A list of datasets.
        separate_eval (bool): Whether to evaluate the results
            separately if it is used as validation dataset.
            Defaults to True.
    """

    def __getitem__(self, index):
        if self.train_cls:
            return self.getitem_cls()
        else:
            return self.getitem()

    def getitem(self):
        valid = False
        while not valid:
            # Select a dataset
            dataset = random.choices(self.datasets, self.prob_datasets)[0]
            data = dataset.prepare_data()

        return data

    def getitem_cls(self):
        pass
