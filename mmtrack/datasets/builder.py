# Copyright (c) OpenMMLab. All rights reserved.
import random
import warnings
from functools import partial

import numpy as np
from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from mmcv.utils import TORCH_VERSION, digit_version
from mmdet.datasets.samplers import (DistributedGroupSampler,
                                     DistributedSampler, GroupSampler)
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

from .base_sot_dataset import BaseSOTDataset
from .samplers import DistributedVideoSampler, SOTVideoSampler


def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     seed=None,
                     persistent_workers=False,
                     **kwargs):
    """Build PyTorch DataLoader.

    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.

    Args:
        dataset (Dataset): A PyTorch dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        seed (int, Optional): Seed to be used. Default: None.
        persistent_workers (bool): If True, the data loader will not shutdown
            the worker processes after a dataset has been consumed once.
            This allows to maintain the workers `Dataset` instances alive.
            This argument is only valid when PyTorch>=1.7.0. Default: False.
        kwargs: any keyword argument to be used to initialize DataLoader

    Returns:
        DataLoader: A PyTorch dataloader.
    """
    rank, world_size = get_dist_info()
    if dist:
        if shuffle:
            sampler = DistributedGroupSampler(dataset, samples_per_gpu,
                                              world_size, rank)
        else:
            if hasattr(dataset, 'load_as_video') and dataset.load_as_video:
                sampler = DistributedVideoSampler(
                    dataset, world_size, rank, shuffle=False)
            else:
                sampler = DistributedSampler(
                    dataset, world_size, rank, shuffle=False)
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        if isinstance(dataset, BaseSOTDataset):
            sampler = SOTVideoSampler(
                dataset) if not shuffle else RandomSampler(dataset)
        else:
            sampler = GroupSampler(dataset,
                                   samples_per_gpu) if shuffle else None
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    if (TORCH_VERSION != 'parrots'
            and digit_version(TORCH_VERSION) >= digit_version('1.7.0')):
        kwargs['persistent_workers'] = persistent_workers
    elif persistent_workers is True:
        warnings.warn('persistent_workers is invalid because your pytorch '
                      'version is lower than 1.7.0')

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        pin_memory=False,
        worker_init_fn=init_fn,
        **kwargs)

    return data_loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
