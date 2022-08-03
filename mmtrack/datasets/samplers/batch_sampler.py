# Copyright (c) OpenMMLab. All rights reserved.
import random
from typing import List, Optional, Sequence

import numpy as np
from torch.utils.data import BatchSampler, Sampler

from mmtrack.registry import DATA_SAMPLERS


@DATA_SAMPLERS.register_module()
class EntireVideoBatchSampler(BatchSampler):
    """A sampler wrapper for grouping images from one video into a same batch.

    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        sampling_frame_range (int): Range of frames sampled from video.
            Defaults to 5.
        sampling_frame_num (int): A specified number of frames.
            Defaults to None.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``.
    """

    def __init__(self,
                 sampler: Sampler,
                 batch_size: int,
                 sampling_frame_range: int = 5,
                 sampling_frame_num: Optional[int] = None,
                 drop_last: bool = False) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError('sampler should be an instance of ``Sampler``, '
                            f'but got {sampler}')

        self.sampler = sampler
        self.batch_size = batch_size
        self.sampling_frame_range = sampling_frame_range
        self.sampling_frame_num = sampling_frame_num
        self.drop_last = drop_last

    def _get_selected_idx(self, video_length: int) -> List[int]:
        ref_frame = random.randrange(video_length)
        start_idx = max(0, ref_frame - self.sampling_frame_range)
        end_idx = min(video_length, ref_frame + self.sampling_frame_range + 1)
        selected_idx = np.random.choice(
            np.array(
                list(range(start_idx, ref_frame)) +
                list(range(ref_frame + 1, end_idx))),
            self.sampling_frame_num - 1,
        )
        selected_idx = selected_idx.tolist() + [ref_frame]
        selected_idx = sorted(selected_idx)
        return selected_idx

    def __iter__(self) -> Sequence[int]:
        batch = []
        for idx in self.sampler:
            data_info = self.sampler.dataset.get_data_info(idx)
            video_length = data_info['video_length']
            batch.append(idx)
            if len(batch) == video_length:
                if self.sampling_frame_num is not None:
                    selected_idx = self._get_selected_idx(video_length - 1)
                    batch = [batch[i] for i in selected_idx]
                yield batch
                batch = []

    def __len__(self) -> int:
        if self.sampler.dataset.test_mode:
            return self.sampler.num_videos
        else:
            if self.drop_last:
                return len(self.sampler) // self.batch_size
            else:
                return (len(self.sampler) + self.batch_size -
                        1) // self.batch_size
