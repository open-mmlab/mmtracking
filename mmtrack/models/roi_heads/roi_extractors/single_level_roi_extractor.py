# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner import force_fp32
from mmdet.models.builder import ROI_EXTRACTORS
from mmdet.models.roi_heads.roi_extractors import \
    SingleRoIExtractor as _SingleRoIExtractor


@ROI_EXTRACTORS.register_module(force=True)
class SingleRoIExtractor(_SingleRoIExtractor):
    """Extract RoI features from a single level feature map.

    This Class is the same as `SingleRoIExtractor` from
    `mmdet.models.roi_heads.roi_extractors` except for using `**kwargs` to
    accept external arguments.
    """

    @force_fp32(apply_to=('feats', ), out_fp16=True)
    def forward(self, feats, rois, roi_scale_factor=None, **kwargs):
        """Forward function."""
        return super().forward(feats, rois, roi_scale_factor)
