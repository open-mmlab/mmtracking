# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
from torch.utils.data import DistributedSampler


class DistributedQuotaSampler(DistributedSampler):
    """Sampler that gets fixed number of samples per epoch.

    Args:
        samples_per_epoch (int): The number of samples per epoch.
    """

    def __init__(self, samples_per_epoch, *args, **kwargs):
        self.samples_per_epoch = samples_per_epoch
        super().__init__(*args, **kwargs)
        self.num_samples = int(
            math.ceil(samples_per_epoch * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed)

        # random sampling `self.samples_per_epoch` samples
        # We only support no-replacement sampling if there are enough samples.
        indices = torch.randperm(len(self.dataset), generator=g)
        if self.samples_per_epoch > len(self.dataset):
            indices = indices.repeat(
                int(math.ceil(self.samples_per_epoch / len(self.dataset))))
        indices = indices[:self.samples_per_epoch]

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)
