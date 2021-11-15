# Copyright (c) OpenMMLab. All rights reserved.
import math

import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info
from torch.utils.data import DistributedSampler, Sampler


class GroupQuotaSampler(Sampler):

    def __init__(self, dataset, samples_per_epoch, samples_per_gpu=1):
        assert hasattr(dataset, 'flag')
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.flag = dataset.flag.astype(np.int64)
        self.samples_per_epoch = samples_per_epoch
        self.group_sizes = np.bincount(self.flag)
        self.num_samples = 0
        self.grounp_sampling_quota = []
        for i, size in enumerate(self.group_sizes):
            # sampling at least 1 samples in each group
            sampling_quota = int(
                np.ceil(samples_per_epoch * size / len(self.dataset)))
            self.grounp_sampling_quota.append(sampling_quota)
            self.num_samples += int(
                np.ceil(sampling_quota /
                        self.samples_per_gpu)) * self.samples_per_gpu

    def __iter__(self):
        indices = []
        for i, size in enumerate(self.group_sizes):
            if size == 0:
                continue
            indice = np.where(self.flag == i)[0]
            assert len(indice) == size
            sampling_size = self.grounp_sampling_quota[i]
            indice = np.random.choice(indice, sampling_size)
            num_extra = int(np.ceil(sampling_size / self.samples_per_gpu)
                            ) * self.samples_per_gpu - len(indice)
            indice = np.concatenate(
                [indice, np.random.choice(indice, num_extra)])
            indices.append(indice)
        indices = np.concatenate(indices)
        indices = [
            indices[i * self.samples_per_gpu:(i + 1) * self.samples_per_gpu]
            for i in np.random.permutation(
                range(len(indices) // self.samples_per_gpu))
        ]
        indices = np.concatenate(indices)
        indices = indices.astype(np.int64).tolist()
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples


class DistributedQuotaSampler(DistributedSampler):
    """Training with fix number samplers during training.

    Args:
        dataset (Dataset): Test dataset that must has `data_infos` attribute.
            Each data_info in `data_infos` record information of one frame,
            and each video must has one data_info that includes
            `data_info['frame_id'] == 0`.
        num_replicas (int): The number of gpus. Defaults to None.
        rank (int): Gpu rank id. Defaults to None.
        shuffle (bool): If True, shuffle the dataset. Defaults to False.
    """

    def __init__(self,
                 samples_per_epoch,
                 num_replicas=None,
                 rank=None,
                 shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError(
                    'Requires distributed package to be available')
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError(
                    'Requires distributed package to be available')
            rank = dist.get_rank()
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.samples_per_epoch = samples_per_epoch
        self.num_samples = int(
            math.ceil(samples_per_epoch * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(
                self.samples_per_epoch, generator=g).tolist()
        else:
            indices = list(range(self.samples_per_epoch))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


class DistributedGroupQuotaSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
        seed (int, optional): random seed used to shuffle the sampler if
            ``shuffle=True``. This number should be identical across all
            processes in the distributed group. Default: 0.
    """

    def __init__(self,
                 dataset,
                 samples_per_epoch,
                 samples_per_gpu=1,
                 num_replicas=None,
                 rank=None,
                 seed=0):
        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.samples_per_epoch = samples_per_epoch
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed if seed is not None else 0

        assert hasattr(self.dataset, 'flag')
        self.flag = self.dataset.flag
        self.group_sizes = np.bincount(self.flag)

        self.num_samples = 0
        self.grounp_sampling_quota = []
        for i, j in enumerate(self.group_sizes):
            # sampling at least 1 samples in each group
            sampling_quota = int(
                math.ceil(samples_per_epoch * self.group_sizes[i] /
                          len(self.dataset)))
            self.grounp_sampling_quota.append(sampling_quota)
            self.num_samples += int(
                math.ceil(sampling_quota * 1.0 / self.samples_per_gpu /
                          self.num_replicas)) * self.samples_per_gpu
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed)

        indices = []
        for i, size in enumerate(self.group_sizes):
            if size > 0:
                indice = np.where(self.flag == i)[0]
                sampling_size = self.grounp_sampling_quota[i]
                assert len(indice) == size
                # add .numpy() to avoid bug when selecting indice in parrots.
                # TODO: check whether torch.randperm() can be replaced by
                # numpy.random.permutation().
                indice = indice[list(
                    torch.randperm(
                        int(size),
                        generator=g).numpy()[:sampling_size])].tolist()
                extra = int(
                    math.ceil(sampling_size * 1.0 / self.samples_per_gpu /
                              self.num_replicas)
                ) * self.samples_per_gpu * self.num_replicas - len(indice)
                # pad indice
                tmp = indice.copy()
                for _ in range(extra // sampling_size):
                    indice.extend(tmp)
                indice.extend(tmp[:extra % sampling_size])
                indices.extend(indice)

        assert len(indices) == self.total_size

        indices = [
            indices[j] for i in list(
                torch.randperm(
                    len(indices) // self.samples_per_gpu, generator=g))
            for j in range(i * self.samples_per_gpu, (i + 1) *
                           self.samples_per_gpu)
        ]

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
