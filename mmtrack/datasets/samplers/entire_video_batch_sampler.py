# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

from torch.utils.data import BatchSampler, Sampler

from mmtrack.registry import DATA_SAMPLERS


@DATA_SAMPLERS.register_module()
class EntireVideoBatchSampler(BatchSampler):
    """A sampler wrapper for grouping images from one video into a same batch.

    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch. Here, we take a video as a batch.
            Defaults to 1.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``. Defaults to False.
    """

    def __init__(self,
                 sampler: Sampler,
                 batch_size: int = 1,
                 drop_last: bool = False) -> None:
        assert sampler.dataset.test_mode
        if not isinstance(sampler, Sampler):
            raise TypeError('sampler should be an instance of ``Sampler``, '
                            f'but got {sampler}')
        if not isinstance(batch_size, int) or batch_size != 1:
            raise ValueError('batch_size should be 1, '
                             f'but got batch_size={batch_size}')
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Sequence[int]:
        batch = []
        for idx in self.sampler:
            data_info = self.sampler.dataset.get_data_info(idx)
            video_length = data_info['video_length']
            batch.append(idx)
            if len(batch) == video_length:
                yield batch
                batch = []

    def __len__(self) -> int:
        return self.sampler.num_videos
