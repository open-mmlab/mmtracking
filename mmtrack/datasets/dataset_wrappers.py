# Copyright (c) OpenMMLab. All rights reserved.
import random

from mmdet.datasets.builder import DATASETS, build_dataset
from torch.utils.data.dataset import ConcatDataset


@DATASETS.register_module()
class RandomSampleConcatDataset(ConcatDataset):
    """A wrapper of concatenated dataset. Support randomly sampling one dataset
    from concatenated datasets and then getting samples from the sampled
    dataset.

    Args:
        dataset_cfgs (list[dict]): The list contains all configs of
            concatenated datasets.
        dataset_sampling_weights (list[float]): The list contains the sampling
            weights of each dataset.
    """

    def __init__(self, dataset_cfgs, dataset_sampling_weights=None):
        if dataset_sampling_weights is None:
            self.dataset_sampling_probs = [1. / len(dataset_cfgs)
                                           ] * len(dataset_cfgs)
        else:
            for x in dataset_sampling_weights:
                assert x >= 0.
            prob_total = float(sum(dataset_sampling_weights))
            assert prob_total > 0.
            self.dataset_sampling_probs = [
                x / prob_total for x in dataset_sampling_weights
            ]

        datasets = [build_dataset(cfg) for cfg in dataset_cfgs]
        # add an attribute `CLASSES` for the calling in `tools/train.py`
        self.CLASSES = datasets[0].CLASSES

        super().__init__(datasets)

    def __getitem__(self, ind):
        """Random sampling a dataset and get samples from this dataset.

        Actually, the input 'ind' is not used in 'dataset'.
        """
        while True:
            dataset = random.choices(self.datasets,
                                     self.dataset_sampling_probs)[0]
            ind = random.randint(0, len(dataset) - 1)
            results = dataset[ind]
            if results is not None:
                return results
