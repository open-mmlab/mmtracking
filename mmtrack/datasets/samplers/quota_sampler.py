# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
from mmcv.runner import get_dist_info
from torch.utils.data import Sampler


class DistributedQuotaSampler(Sampler):
    """Sampler that gets fixed number of samples per epoch.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Args:
        dataset: Dataset used for sampling.
        samples_per_epoch (int): The number of samples per epoch.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
        replacement (bool): samples are drawn with replacement if ``True``,
            Default: False.
        seed (int, optional): random seed used to shuffle the sampler if
            ``shuffle=True``. This number should be identical across all
            processes in the distributed group. Default: 0.
    """

    def __init__(self,
                 dataset,
                 samples_per_epoch,
                 num_replicas=None,
                 rank=None,
                 replacement=False,
                 seed=0):
        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank
        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed if seed is not None else 0
        self.replacement = replacement

        self.num_samples = int(
            math.ceil(samples_per_epoch * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
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
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
