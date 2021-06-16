from mmcv.parallel import collate
from collections.abc import Mapping, Sequence


def collate_enhance(batch, samples_per_gpu=1):

    if not isinstance(batch, Sequence):
        raise TypeError(f'{batch.dtype} is not supported.')

    if isinstance(batch[0], Sequence) and isinstance(batch[0][0], Mapping):
        assert samples_per_gpu == 1
        batch = batch[0]
        samples_per_gpu = len(batch)

    return collate(batch, samples_per_gpu)