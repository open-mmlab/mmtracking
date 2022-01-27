# Copyright (c) OpenMMLab. All rights reserved.
import random

from torch.utils.data.dataset import ConcatDataset
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.builder import build_dataset


@DATASETS.register_module()
class RandomSampleConcatDataset(ConcatDataset):
    """A wrapper of concatenated dataset. Support randomly sampling one dataset
        from concatenated datasets and then getting samples from the sampled
        dataset.

    Args:
        dataset_cfgs (list[dict]): The list contains all configs of
            concatenated datasets.
        dataset_sampling_weights (list): The list contains the sampling
            weights of each dataset.
    """

    def __init__(self, dataset_cfgs, dataset_sampling_weights=None):
        datasets = [build_dataset(cfg) for cfg in dataset_cfgs]
        super().__init__(datasets)
        if dataset_sampling_weights is None:
            self.dataset_sampling_probs = [1 / len(datasets)] * len(datasets)
        else:
            prob_total = sum(dataset_sampling_weights)
            self.dataset_sampling_probs = [
                x / prob_total for x in dataset_sampling_weights
            ]

    def __getitem__(self, ind):
        """Random sampling a dataset and get samples from this dataset.
            Actually, the input 'ind' is not used in 'dataset'.
        """
        while True:
            dataset = random.choices(self.datasets,
                                     self.dataset_sampling_probs)[0]
            results = dataset[ind]
            if results is not None:
                return results
