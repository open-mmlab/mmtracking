# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.ops import PrRoIPool
from mmengine.model import BaseModule
from torch import Tensor, nn

from mmtrack.registry import MODELS


@MODELS.register_module()
class FilterInitializer(BaseModule):
    """Initializes a target classification filter.

    Args:
        filter_size (int, optional):  Size of the filter. Defaults to 4.
        feature_dim (int, optional):  Input feature dimentionality.
             Defaults to 512.
        feature_stride (int, optional):  Input feature stride. Defaults to 16.
    """

    def __init__(self,
                 filter_size: int = 4,
                 feature_dim: int = 512,
                 feature_stride: int = 16):
        super().__init__()
        self.filter_conv = nn.Conv2d(
            feature_dim, feature_dim, kernel_size=3, padding=1)
        self.filter_pool = PrRoIPool(filter_size, 1 / feature_stride)

    def init_weights(self):
        """Initialize the parameters of this module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, feat: Tensor, bboxes: Tensor) -> Tensor:
        """Runs the initializer module. Note that [] denotes an optional
        dimension.

        Args:
            feat (Tensor):  Input feature maps with shape
                (images_in_sequence, [sequences], feat_dim, H, W).
            bboxes (Tensor):  Target bounding boxes with
                (images_in_sequence, [sequences], 4) shape in [x1, y1, x2, y2]
                format.

        Returns:
            filter_weights (Tensor):  The output filter with shape
                (images_in_sequence, c, filter_h, filter_w).
        """

        num_images = feat.shape[0]

        feat = self.filter_conv(feat.reshape(-1, *feat.shape[-3:]))

        # Add batch_index to rois
        batch_index = torch.arange(
            bboxes.shape[0], dtype=torch.float32).reshape(-1,
                                                          1).to(bboxes.device)
        roi = torch.cat((batch_index, bboxes), dim=1)
        filter_weights = self.filter_pool(feat, roi)

        # If multiple input images, compute the initial filter
        # as the average filter.
        if num_images > 1:
            filter_weights = torch.mean(
                filter_weights.reshape(num_images, -1,
                                       *filter_weights.shape[-3:]),
                dim=0)

        return filter_weights
