# Copyright (c) OpenMMLab. All rights reserved.
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
            window = window.flatten().repeat(self.num_base_anchors[i])
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
            feature map in [tl_x, tl_y, br_x, br_y] format.
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
            ws = ((w * w_ratios[None, :]).long() * scales[:, None]).view(-1)
            hs = ((h * h_ratios[None, :]).long() * scales[:, None]).view(-1)

        # use float anchor and the anchor's center is aligned with the
        # pixel point
        base_anchors = [
            x_center - 0.5 * ws, y_center - 0.5 * hs, x_center + 0.5 * ws,
            y_center + 0.5 * hs
        ]
        base_anchors = torch.stack(base_anchors, dim=-1)

        return base_anchors
