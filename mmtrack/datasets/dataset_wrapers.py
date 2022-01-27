# Copyright (c) OpenMMLab. All rights reserved.
import random

from torch.utils.data.dataset import ConcatDataset
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.builder import build_dataset


@DATASETS.register_module()
class RandomSampleConcatDataset(ConcatDataset):
    """A wrapper of concatenated dataset. Support randomly sampling one dataset
        from concatenated datasets.

    Args:
        dataset_cfgs (list[dict]): The list contains the configs of several
            datasets.
        datasets_sampling_prob (list): The list contains the sampling
            probilities of each dataset.
    """

    def __init__(self, dataset_cfgs, datasets_sampling_prob=None):
        datasets = [build_dataset(cfg) for cfg in dataset_cfgs]
        super().__init__(datasets)
        if datasets_sampling_prob is None:
            self.datasets_sampling_prob = [1 / len(datasets)] * len(datasets)
        else:
            prob_total = sum(datasets_sampling_prob)
            self.datasets_sampling_prob = [
                x / prob_total for x in datasets_sampling_prob
            ]

    def __getitem__(self, ind):
        while True:
            dataset = random.choices(self.datasets,
                                     self.datasets_sampling_prob)[0]
            results = dataset[ind]
            if results is not None:
                return results
