# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcls.models.builder import NECKS
from mmcls.models.necks import GlobalAveragePooling as _GlobalAveragePooling


@NECKS.register_module(force=True)
class GlobalAveragePooling(_GlobalAveragePooling):
    """Global Average Pooling neck.

    Note that we use `view` to remove extra channel after pooling. We do not
    use `squeeze` as it will also remove the batch dimension when the tensor
    has a batch dimension of size 1, which can lead to unexpected errors.
    """

    def __init__(self, kernel_size=None, stride=None):
        super(GlobalAveragePooling, self).__init__()
        if kernel_size is None and stride is None:
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.gap = nn.AvgPool2d(kernel_size, stride)
