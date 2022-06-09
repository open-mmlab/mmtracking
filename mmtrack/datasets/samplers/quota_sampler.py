# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Iterator, Sized

import torch
from mmengine.dist import get_dist_info
from torch.utils.data import Sampler

from mmtrack.registry import DATA_SAMPLERS


@DATA_SAMPLERS.register_module()
class QuotaSampler(Sampler):
    """Sampler that gets fixed number of samples per epoch.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Args:
        dataset (Sized): Dataset used for sampling.
        samples_per_epoch (int): The number of samples per epoch.
        replacement (bool): samples are drawn with replacement if ``True``,
            Default: False.
        seed (int, optional): random seed used to shuffle the sampler if
            ``shuffle=True``. This number should be identical across all
            processes in the distributed group. Default: 0.
    """

    def __init__(self,
                 dataset: Sized,
                 samples_per_epoch: int,
                 replacement: bool = False,
                 seed: int = 0) -> None:
        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size

        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch
        self.epoch = 0
        self.seed = seed if seed is not None else 0
        self.replacement = replacement

        self.num_samples = int(
            math.ceil(samples_per_epoch * 1.0 / self.world_size))
        self.total_size = self.num_samples * self.world_size

    def __iter__(self) -> Iterator:
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed)

        # random sampling `self.samples_per_epoch` samples
        if self.replacement:
            indices = torch.randint(
                len(self.dataset),
                size=(self.samples_per_epoch, ),
                dtype=torch.int64).tolist()
        else:
            indices = torch.randperm(len(self.dataset), generator=g)
            if self.samples_per_epoch > len(self.dataset):
                indices = indices.repeat(
                    int(math.ceil(self.samples_per_epoch / len(self.dataset))))
            indices = indices[:self.samples_per_epoch].tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.world_size]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
