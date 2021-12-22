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

    def __init__(self,
                 datasets,
                 datasets_sampling_prob=None,
                 train_cls=False,
                 separate_eval=True):
        super().__init__(datasets, separate_eval=separate_eval)
        if datasets_sampling_prob is None:
            self.datasets_sampling_prob = [1 / len(datasets)] * len(datasets)
        else:
            prob_total = sum(datasets_sampling_prob)
            self.datasets_sampling_prob = [
                x / prob_total for x in datasets_sampling_prob
            ]
        self.train_cls = train_cls

    def __getitem__(self, idx):
        dataset = random.choices(self.datasets, self.datasets_sampling_prob)[0]
        if self.train_cls:
            return dataset.prepare_train_cls_img(idx)
        else:
            return dataset.prepare_train_reg_img(idx)
