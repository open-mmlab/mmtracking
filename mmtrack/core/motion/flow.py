# Copyright (c) OpenMMLab. All rights reserved.
import torch


def flow_warp_feats(x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Use flow to warp feature map.

    Args:
        x (Tensor): of shape (N, C, H_x, W_x).
        flow (Tensor): of shape (N, C, H_f, W_f).

    Returns:
        Tensor: The warpped feature map with shape (N, C, H_x, W_x).
    """
    assert x.dim() == 4
    assert flow.dim() == 4 and flow.size(1) == 2
    # 1. resize the resolution of flow to be the same as x.
    scale_factor_w = float(x.shape[-1]) / flow.shape[-1]
    scale_factor_h = float(x.shape[-2]) / flow.shape[-2]
    flow = torch.nn.functional.interpolate(
        flow, size=x.shape[-2:], mode='bilinear', align_corners=False)
    flow[:, 0] = flow[:, 0] * scale_factor_w
    flow[:, 1] = flow[:, 1] * scale_factor_h

    # 2. compute the flow_field (grid in the code) used to warp features.
    H, W = x.shape[-2:]
    h_grid, w_grid = torch.meshgrid(torch.arange(H), torch.arange(W))
    # [1, 1, H, W]
    h_grid = h_grid.to(flow)[None, None, ...]
    # [1, 1, H, W]
    w_grid = w_grid.to(flow)[None, None, ...]
    # [1, 2, H, W]
    grid = torch.cat((w_grid, h_grid), dim=1)
    # [N, 2, H, W]
    grid = grid + flow
    grid[:, 0] = grid[:, 0] / W * 2 - 1
    grid[:, 1] = grid[:, 1] / H * 2 - 1
    # [N, H, W, 2]
    grid = grid.permute(0, 2, 3, 1)

    # 3. warp features.
    x_warp = torch.nn.functional.grid_sample(
        x, grid, padding_mode='border', align_corners=True)
    return x_warp
