import torch
from mmcv.runner import force_fp32

from mmdet.models.builder import ROI_EXTRACTORS
from mmdet.models.roi_heads import SingleRoIExtractor


@ROI_EXTRACTORS.register_module()
class CenterRoIExtractor(SingleRoIExtractor):
    """Extract RoI features from a single level feature map.

    If there are multiple input feature levels, each RoI is mapped to a level
    according to its scale. The mapping rule is proposed in
    `FPN <https://arxiv.org/abs/1612.03144>`_.

    Args:
        roi_layer (dict): Specify RoI layer type and arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (int): Strides of input feature maps.
        finest_scale (int): Scale threshold of mapping to level 0. Default: 56.
    """

    def __init__(self, roi_scale_factor=-1, *args, **kwargs):
        super(CenterRoIExtractor, self).__init__(*args, **kwargs)
        self.roi_scale_factor = roi_scale_factor

    def roi_rescale(self, rois, scale_factor):
        """Scale RoI coordinates by scale factor.

        Args:
            rois (torch.Tensor): RoI (Region of Interest), shape (n, 5)
            scale_factor (float): Scale factor that RoI will be multiplied by.

        Returns:
            torch.Tensor: Scaled RoI.
        """

        cx = (rois[:, 1] + rois[:, 3]) * 0.5
        # cy = (rois[:, 2] + rois[:, 4]) * 0.5
        w = rois[:, 3] - rois[:, 1]
        # h = rois[:, 4] - rois[:, 2]
        new_w = w * scale_factor
        # new_h = h * scale_factor
        x1 = cx - new_w * 0.5
        x2 = cx + new_w * 0.5
        # y1 = cy - new_h * 0.5
        # y2 = cy + new_h * 0.5
        rois[:, 1] = x1
        rois[:, 3] = x2
        return rois

    @force_fp32(apply_to=('feats', ), out_fp16=True)
    def forward(self, feats, rois):
        """Forward function."""
        out_size = self.roi_layers[0].output_size
        num_levels = len(feats)
        roi_feats = feats[0].new_zeros(
            rois.size(0), self.out_channels, *out_size)
        # TODO: remove this when parrots supports
        if torch.__version__ == 'parrots':
            roi_feats.requires_grad = True

        if num_levels == 1:
            if len(rois) == 0:
                return roi_feats
            if self.roi_scale_factor > 0:
                rois = self.roi_rescale(rois, self.roi_scale_factor)
            return self.roi_layers[0](feats[0], rois)

        target_lvls = self.map_roi_levels(rois, num_levels)
        if self.roi_scale_factor > 0:
            rois = self.roi_rescale(rois, self.roi_scale_factor)
        for i in range(num_levels):
            inds = target_lvls == i
            if inds.any():
                rois_ = rois[inds, :]
                roi_feats_t = self.roi_layers[i](feats[i], rois_)
                roi_feats[inds] = roi_feats_t
            else:
                roi_feats += sum(
                    x.view(-1)[0]
                    for x in self.parameters()) * 0. + feats[i].sum() * 0.
        return roi_feats
