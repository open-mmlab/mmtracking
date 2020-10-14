import torch


def flow_warp_feats(x, flow):
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
    h_grid = torch.tensor(h_grid, device=flow.device).float()[None, None, ...]
    # [1, 1, H, W]
    w_grid = torch.tensor(w_grid, device=flow.device).float()[None, None, ...]
    # [1, 2, H, W]
    grid = torch.cat((w_grid, h_grid), dim=1)
    # [N, 2, H, W]
    grid = grid + flow
    grid[:, 0] = grid[:, 0] / W * 2 - 1
    grid[:, 1] = grid[:, 1] / H * 2 - 1
    # [N, H, W, 2]
    grid = grid.permute(0, 2, 3, 1)

    # 3. warp features.
    x = torch.nn.functional.grid_sample(
        x, grid, padding_mode='border', align_corners=False)
    return x
