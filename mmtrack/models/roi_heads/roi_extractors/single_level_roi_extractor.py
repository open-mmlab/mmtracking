from mmcv.runner import force_fp32
from mmdet.models.builder import ROI_EXTRACTORS
from mmdet.models.roi_heads.roi_extractors import \
    SingleRoIExtractor as _SingleRoIExtractor


@ROI_EXTRACTORS.register_module(force=True)
class SingleRoIExtractor(_SingleRoIExtractor):

    @force_fp32(apply_to=('feats', ), out_fp16=True)
    def forward(self, feats, rois, roi_scale_factor=None, **kwargs):
        return super().forward(feats, rois, roi_scale_factor)
