import torch


def flow_warp_feats(flow, ref_x_single):
    # 1. resize the resolution of flow to be the same as ref_x_single.
    scale_factor = float(ref_x_single.shape[-1]) / flow.shape[-1]
    flow = torch.nn.functional.interpolate(
        flow, scale_factor=scale_factor, mode='bilinear', align_corners=False)
    flow = flow * scale_factor

    # 2. compute the flow_field (grid in the code) used to warp features.
    H, W = ref_x_single.shape[-2:]
    h_grid, w_grid = torch.meshgrid(torch.arange(H), torch.arange(W))
    # [1, 1, H, W]
    h_grid = h_grid.float().cuda()[None, None, ...]
    # [1, 1, H, W]
    w_grid = w_grid.float().cuda()[None, None, ...]
    # [1, 2, H, W]
    grid = torch.cat((w_grid, h_grid), dim=1)
    grid = grid + flow
    grid[:, 0] = grid[:, 0] / W * 2 - 1
    grid[:, 1] = grid[:, 1] / H * 2 - 1
    # [1, H, W, 2]
    grid = grid.permute(0, 2, 3, 1)

    # 3. warp features.
    x_single = torch.nn.functional.grid_sample(
        ref_x_single, grid, padding_mode='border', align_corners=False)
    return x_single
