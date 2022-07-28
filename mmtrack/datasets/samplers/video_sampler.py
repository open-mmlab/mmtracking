# Copyright (c) OpenMMLab. All rights reserved.
from typing import Iterator, Sized

import numpy as np
from mmengine.dist import get_dist_info
from torch.utils.data import Sampler

from mmtrack.datasets import BaseSOTDataset, BaseVideoDataset
from mmtrack.registry import DATA_SAMPLERS


@DATA_SAMPLERS.register_module()
class VideoSampler(Sampler):
    """The video data sampler is for both distributed and non-distributed
    environment. It is only used in testing.

    Args:
        dataset (Sized): The dataset.
    """

    def __init__(self, dataset: Sized, seed: int = 0) -> None:
        self.dataset = dataset
        assert self.dataset.test_mode

        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size

        if isinstance(self.dataset, BaseSOTDataset):
            # The input of '__getitem__' function in SOT dataset class must be
            # a tuple when testing. The tuple is in (video_index, frame_index)
            # format.
            self.num_videos = self.dataset.num_videos
            if self.num_videos < self.world_size:
                raise ValueError(f'only {self.num_videos} videos loaded,'
                                 f'but {self.world_size} gpus were given.')

            chunks = np.array_split(
                list(range(self.num_videos)), self.world_size)
            self.indices = []
            for videos in chunks:
                indices_chunk = []
                for video_ind in videos:
                    indices_chunk.extend([
                        (video_ind, frame_ind) for frame_ind in range(
                            self.dataset.get_len_per_video(video_ind))
                    ])
                self.indices.append(indices_chunk)
        else:
            assert isinstance(self.dataset, BaseVideoDataset)
            first_frame_indices = []
            for i in range(len(self.dataset)):
                data_info = self.dataset.get_data_info(i)
                if data_info['frame_id'] == 0:
                    first_frame_indices.append(i)

            self.num_videos = len(first_frame_indices)
            if self.num_videos < self.world_size:
                raise ValueError(f'only {self.num_videos} videos loaded,'
                                 f'but {self.world_size} gpus were given.')

            chunks = np.array_split(first_frame_indices, self.world_size)
            split_flags = [c[0] for c in chunks]
            split_flags.append(len(self.dataset))

            self.indices = [
                list(range(split_flags[i], split_flags[i + 1]))
                for i in range(self.world_size)
            ]

    def __iter__(self) -> Iterator[int]:
        """Iterate the indices."""
        indices = self.indices[self.rank]
        return iter(indices)

    def __len__(self) -> int:
        """The number of samples in this rank."""
        return len(self.indices[self.rank])

    def set_epoch(self, epoch: int) -> None:
        """Not supported in iteration-based runner."""
        raise NotImplementedError(
            'The `VideoSampler` is only used in testing, '
            "and doesn't need `set_epoch`")
