# Copyright (c) OpenMMLab. All rights reserved.
import random
from typing import List, Optional

from mmengine.dataset import ConcatDataset

from mmtrack.registry import DATASETS


@DATASETS.register_module()
class RandomSampleConcatDataset(ConcatDataset):
    """A wrapper of concatenated dataset. Support randomly sampling one dataset
    from concatenated datasets and then getting samples from the sampled
    dataset. This class only support training.

    Args:
        datasets (list[dict]): The list contains all configs of
            concatenated datasets.
        dataset_sampling_weights (Optional[List[float]], optional): The list
            contains the sampling weights of each dataset. Defaults to None.
    """

    def __init__(self,
                 datasets: List[dict],
                 dataset_sampling_weights: Optional[List[float]] = None):
        if dataset_sampling_weights is None:
            self.dataset_sampling_probs = [1. / len(datasets)] * len(datasets)
        else:
            for x in dataset_sampling_weights:
                assert x >= 0.
            prob_total = float(sum(dataset_sampling_weights))
            assert prob_total > 0.
            self.dataset_sampling_probs = [
                x / prob_total for x in dataset_sampling_weights
            ]

        datasets = [DATASETS.build(cfg) for cfg in datasets]
        # add an attribute `classes` for the calling in `tools/train.py`
        self.classes = datasets[0].META['classes']

        super().__init__(datasets)

    def __getitem__(self, ind: int) -> dict:
        """Random sampling a dataset and get samples from this dataset..

        Args:
            ind (int): The random index.  Actually, in this class,
                the input 'ind' is not used in 'dataset'.

        Returns:
            dict: The results after the dataset pipeline.
        """
        while True:
            dataset = random.choices(self.datasets,
                                     self.dataset_sampling_probs)[0]
            ind = random.randint(0, len(dataset) - 1)
            results = dataset[ind]
            if results is not None:
                return results
