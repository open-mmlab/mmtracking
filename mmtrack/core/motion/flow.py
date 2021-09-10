# Copyright (c) OpenMMLab. All rights reserved.
import torch


def flow_warp_feats(x, flow):
    """Use flow to warp feature map.

    Args:
        x (Tensor): of shape (N, C, H_x, W_x).
        flow (Tensor): of shape (N, C, H_f, W_f).

    Returns:
        Tensor: The warpped feature map with shape (N, C, H_x, W_x).
    """
    assert len(x.shape) == 4
    assert len(flow.shape) == 4 and flow.shape[1] == 2
    # 1. resize the resolution of flow to be the same as x.
    scale_factor = float(x.shape[-1]) / flow.shape[-1]
    flow = torch.nn.functional.interpolate(
        flow, scale_factor=scale_factor, mode='bilinear', align_corners=False)
    flow = flow * scale_factor

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
