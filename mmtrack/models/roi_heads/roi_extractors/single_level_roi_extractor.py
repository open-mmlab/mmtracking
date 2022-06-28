# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

from mmdet.models.roi_heads.roi_extractors import \
    SingleRoIExtractor as _SingleRoIExtractor
from torch import Tensor

from mmtrack.registry import MODELS


@MODELS.register_module()
class SingleRoIExtractor(_SingleRoIExtractor):
    """Extract RoI features from a single level feature map.

    This Class is the same as `SingleRoIExtractor` from
    `mmdet.models.roi_heads.roi_extractors` except for using `**kwargs` to
    accept external arguments.
    """

    def forward(self,
                feats: Tuple[Tensor],
                rois: Tensor,
                roi_scale_factor: float = None,
                **kwargs) -> Tensor:
        """Forward function.
        Args:
            feats (Tuple[Tensor]): The feature maps.
            rois (Tensor): The RoIs.
            roi_scale_factor (float): Scale factor that RoI will be multiplied
                by.

        Returns:
            Tensor: RoI features.
        """
        return super().forward(feats, rois, roi_scale_factor)
