# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from torch.utils.data import DistributedSampler as _DistributedSampler
from torch.utils.data import Sampler

from mmtrack.datasets.base_sot_dataset import BaseSOTDataset


class SOTVideoSampler(Sampler):
    """Only used for sot testing on single gpu.

    Args:
        dataset (Dataset): Test dataset must have `num_frames_per_video`
            attribute. It records the frame number of each video.
    """

    def __init__(self, dataset):
        super().__init__(dataset)
        # The input of '__getitem__' function in SOT dataset class must be
        # a tuple when testing. The tuple is in (video_index, frame_index)
        # format.
        self.dataset = dataset
        self.indices = []
        for video_ind, num_frames in enumerate(
                self.dataset.num_frames_per_video):
            self.indices.extend([(video_ind, frame_ind)
                                 for frame_ind in range(num_frames)])

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.dataset)


class DistributedVideoSampler(_DistributedSampler):
    """Put videos to multi gpus during testing.

    Args:
        dataset (Dataset): Test dataset must have `data_infos` attribute.
            Each data_info in `data_infos` records information of one frame or
            one video (in SOT Dataset). If not SOT Dataset, each video must
            have one data_info that includes `data_info['frame_id'] == 0`.
        num_replicas (int): The number of gpus. Defaults to None.
        rank (int): Gpu rank id. Defaults to None.
        shuffle (bool): If True, shuffle the dataset. Defaults to False.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle
        assert not self.shuffle, 'Specific for video sequential testing.'
        self.num_samples = len(dataset)

        if isinstance(dataset, BaseSOTDataset):
            # The input of '__getitem__' function in SOT dataset class must be
            # a tuple when testing. The tuple is in (video_index, frame_index)
            # format.
            self.num_videos = len(self.dataset.data_infos)
            self.num_frames_per_video = self.dataset.num_frames_per_video
            if self.num_videos < num_replicas:
                raise ValueError(f'only {self.num_videos} videos loaded,'
                                 f'but {self.num_replicas} gpus were given.')

            chunks = np.array_split(
                list(range(self.num_videos)), self.num_replicas)
            self.indices = []
            for videos in chunks:
                indices_chunk = []
                for video_ind in videos:
                    indices_chunk.extend([
                        (video_ind, frame_ind) for frame_ind in range(
                            self.num_frames_per_video[video_ind])
                    ])
                self.indices.append(indices_chunk)
        else:
            first_frame_indices = []
            for i, img_info in enumerate(self.dataset.data_infos):
                if img_info['frame_id'] == 0:
                    first_frame_indices.append(i)

            if len(first_frame_indices) < num_replicas:
                raise ValueError(
                    f'only {len(first_frame_indices)} videos loaded,'
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
