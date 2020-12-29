import numpy as np
import torch
from mmdet.core.anchor import ANCHOR_GENERATORS, AnchorGenerator


@ANCHOR_GENERATORS.register_module()
class SiameseRPNAnchorGenerator(AnchorGenerator):
    """Anchor generator for siamese rpn.

    Please refer to `mmdet/core/anchor/anchor_generator.py:AnchorGenerator`
    for detailed docstring.
    """

    def __init__(self, strides, *args, **kwargs):
        assert len(strides) == 1, 'only support one feature map level'
        super(SiameseRPNAnchorGenerator,
              self).__init__(strides, *args, **kwargs)

    def gen_2d_hanning_windows(self, featmap_sizes, device='cuda'):
        """Generate 2D hanning window.

        Args:
            featmap_sizes (list[torch.size]): List of torch.size recording the
                resolution (height, width) of the multi-level feature maps.
            device (str): Device the tensor will be put on. Defaults to 'cuda'.

        Returns:
            list[Tensor]: List of 2D hanning window with shape
            (num_base_anchors[i] * featmap_sizes[i][0] * featmap_sizes[i][1]).
        """
        assert self.num_levels == len(featmap_sizes)
        multi_level_windows = []
        for i in range(self.num_levels):
            hanning_h = np.hanning(featmap_sizes[i][0])
            hanning_w = np.hanning(featmap_sizes[i][1])
            window = np.outer(hanning_h, hanning_w)
            window = np.tile(window.flatten(), self.num_base_anchors[i])
            multi_level_windows.append(torch.from_numpy(window).to(device))
        return multi_level_windows

    def gen_single_level_base_anchors(self,
                                      base_size,
                                      scales,
                                      ratios,
                                      center=None):
        """Generate base anchors of a single level feature map.

        Args:
            base_size (int | float): Basic size of an anchor.
            scales (torch.Tensor): Scales of the anchor.
            ratios (torch.Tensor): The ratio between between the height
                and width of anchors in a single level.
            center (tuple[float], optional): The center of the base anchor
                related to a single feature grid. Defaults to None.

        Returns:
            torch.Tensor: Anchors of one spatial location in a single level
            feature map in [cx, cy, w, h] format.
        """
        w = base_size
        h = base_size
        if center is None:
            x_center = self.center_offset * w
            y_center = self.center_offset * h
        else:
            x_center, y_center = center

        h_ratios = torch.sqrt(ratios)
        w_ratios = 1 / h_ratios
        if self.scale_major:
            ws = ((w * w_ratios[:, None]).long() * scales[None, :]).view(-1)
            hs = ((h * h_ratios[:, None]).long() * scales[None, :]).view(-1)
        else:
            ws = (w * scales[:, None] * w_ratios[None, :]).view(-1)
            hs = (h * scales[:, None] * h_ratios[None, :]).view(-1)

        base_anchors = [
            torch.ones_like(ws) * x_center,
            torch.ones_like(hs) * y_center, ws, hs
        ]
        base_anchors = torch.stack(base_anchors, dim=-1)

        return base_anchors

    def single_level_grid_anchors(self,
                                  base_anchors,
                                  featmap_size,
                                  stride=(16, 16),
                                  device='cuda'):
        """Generate grid anchors of a single level feature map.

        Note:
            This function is usually called by method ``self.grid_anchors``.

        Args:
            base_anchors (torch.Tensor): The base anchors of a feature grid.
            featmap_size (tuple[int]): Size of the feature maps in order
                (h, w).
            stride (tuple[int], optional): Stride of the feature map in order
                (w, h). Defaults to (16, 16).
            device (str, optional): Device the tensor will be put on.
                Defaults to 'cuda'.

        Returns:
            torch.Tensor: Anchors of all spatial locations with [cx, cy, w, h]
            format in the feature map.
        """
        feat_h, feat_w = featmap_size
        # convert Tensor to int, so that we can covert to ONNX correctlly
        feat_h = int(feat_h)
        feat_w = int(feat_w)
        shift_x = torch.arange(0, feat_w, device=device) * stride[0]
        shift_y = torch.arange(0, feat_h, device=device) * stride[1]

        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = torch.stack([
            shift_xx, shift_yy,
            torch.zeros_like(shift_xx),
            torch.zeros_like(shift_yy)
        ],
                             dim=-1)
        shifts = shifts.type_as(base_anchors)

        all_anchors = base_anchors[:, None, :] + shifts[None, :, :]
        all_anchors = all_anchors.view(-1, 4)

        all_anchors[:, 0] += -(feat_w // 2) * stride[0]
        all_anchors[:, 1] += -(feat_h // 2) * stride[1]

        return all_anchors
