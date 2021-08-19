# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from torch.utils.data import DistributedSampler as _DistributedSampler


class DistributedVideoSampler(_DistributedSampler):
    """Put videos to multi gpus during testing.

    Args:
        dataset (Dataset): Test dataset that must has `data_infos` attribute.
            Each data_info in `data_infos` record information of one frame,
            and each video must has one data_info that includes
            `data_info['frame_id'] == 0`.
        num_replicas (int): The number of gpus. Defaults to None.
        rank (int): Gpu rank id. Defaults to None.
        shuffle (bool): If True, shuffle the dataset. Defaults to False.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle
        assert not self.shuffle, 'Specific for video sequential testing.'
        self.num_samples = len(dataset)

        first_frame_indices = []
        for i, img_info in enumerate(self.dataset.data_infos):
            if img_info['frame_id'] == 0:
                first_frame_indices.append(i)

        if len(first_frame_indices) < num_replicas:
            raise ValueError(f'only {len(first_frame_indices)} videos loaded,'
                             f'but {self.num_replicas} gpus were given.')

        chunks = np.array_split(first_frame_indices, self.num_replicas)
        split_flags = [c[0] for c in chunks]
        split_flags.append(self.num_samples)

        self.indices = [
            list(range(split_flags[i], split_flags[i + 1]))
            for i in range(self.num_replicas)
        ]

    def __iter__(self):
        """Put videos to specify gpu."""
        indices = self.indices[self.rank]
        return iter(indices)
