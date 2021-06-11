import copy
import math
from collections import defaultdict
import random

import numpy as np
from mmcv.runner import get_dist_info
from torch.utils.data.sampler import Sampler

class IdentitySampler(Sampler):
    """Randomly samples N identities each with K instances.

    Args:
        dataset (list): contains tuples of (img_path(s), pid, camid, dsetid).
        samples_per_gpu (int): batch size.
        num_instances (int): number of instances per identity in a batch.
    """

    def __init__(self,
                 dataset,
                 samples_per_gpu,
                 num_instances):
        if samples_per_gpu < num_instances:
            raise ValueError(
                'batch_size={} must be no less '
                'than num_instances={}'.format(samples_per_gpu, num_instances)
            )

        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.num_instances = num_instances
        self.num_pids_per_batch = self.samples_per_gpu // self.num_instances
        self.index_dic = defaultdict(list)
        for index, infos in enumerate(dataset.data_infos):
            pid = infos['gt_label']
            self.index_dic[int(pid)].append(index)
        self.pids = list(self.index_dic.keys())
        assert len(self.pids) >= self.num_pids_per_batch

        # estimate number of examples in an epoch
        # TODO: improve precision
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(
                    idxs, size=self.num_instances, replace=True
                )
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []
        cur_length = 0
        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                cur_length += len(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)
        self.length = cur_length

        return iter(final_idxs)

    def __len__(self):
        return self.length

class DistributedIdentitySampler(Sampler):
    """Randomly samples N identities each with K instances.

    Args:
        dataset (list): Dataset used for sampling.
        samples_per_gpu (int): batch size.
        num_instances (int): number of instances per identity in a batch.
    """

    def __init__(self,
                 dataset,
                 samples_per_gpu,
                 num_instances,
                 num_replicas=None,
                 rank=None,
                 seed=0):
        if samples_per_gpu < num_instances:
            raise ValueError(
                'batch_size={} must be no less '
                'than num_instances={}'.format(samples_per_gpu, num_instances)
            )
        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.num_instances = num_instances
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed if seed is not None else 0

        self.num_pids_per_batch = self.samples_per_gpu // self.num_instances
        self.index_dic = defaultdict(list)
        for index, infos in enumerate(dataset.data_infos):
            pid = infos['gt_label']
            self.index_dic[int(pid)].append(index)
        self.pids = list(self.index_dic.keys())
        assert len(self.pids) >= self.num_pids_per_batch

        # estimate number of examples in an epoch
        self.length = 0

    def __iter__(self):
        print(self.epoch)
        np.random.seed(self.epoch + self.seed)
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(
                    idxs, size=self.num_instances, replace=True
                )
            np.random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []
        cur_length = 0
        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                cur_length += len(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)
        self.length = cur_length

        self.num_samples = math.ceil((self.length - self.num_replicas) / self.num_replicas)

        # subsample
        offset = self.num_samples * self.rank
        final_idxs = final_idxs[offset:offset + self.num_samples]
        assert len(final_idxs) == self.num_samples

        return iter(final_idxs)

    def __len__(self):
        return self.length

    def set_epoch(self, epoch):
        self.epoch = epoch