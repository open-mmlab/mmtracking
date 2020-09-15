import numpy as np
from torch.utils.data import DistributedSampler as _DistributedSampler


class DistributedVideoSampler(_DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle
        assert not self.shuffle, 'Specific for video sequential testing.'
        self.num_samples = len(dataset)

        first_frame_indices = []
        for i, img_info in enumerate(self.dataset.data_infos):
            if img_info['frame_id'] == 0:
                first_frame_indices.append(i)

        chunks = np.array_split(first_frame_indices, num_replicas)
        split_flags = [c[0] for c in chunks]
        split_flags.append(self.num_samples)

        self.indices = [
            list(range(split_flags[i], split_flags[i + 1]))
            for i in range(self.num_replicas)
        ]

    def __iter__(self):
        indices = self.indices[self.rank]
        return iter(indices)
