# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmtrack.models.backbones import ConvVisionTransformer


def test_sot_ConvVisionTransformer():
    """Test MixFormer CVT backbone."""

    cfg = dict(
        num_stages=3,
        patch_size=[7, 3, 3],
        patch_stride=[4, 2, 2],
        patch_padding=[2, 1, 1],
        dim_embed=[64, 192, 384],
        num_heads=[1, 3, 6],
        depth=[1, 4, 16],
        mlp_channel_ratio=[4, 4, 4],
        attn_drop_rate=[0.0, 0.0, 0.0],
        drop_rate=[0.0, 0.0, 0.0],
        path_drop_probs=[0.0, 0.0, 0.1],
        qkv_bias=[True, True, True],
        qkv_proj_method=['dw_bn', 'dw_bn', 'dw_bn'],
        kernel_qkv=[3, 3, 3],
        padding_kv=[1, 1, 1],
        stride_kv=[2, 2, 2],
        padding_q=[1, 1, 1],
        stride_q=[1, 1, 1],
        norm_cfg=dict(type='BN', requires_grad=False))

    model = ConvVisionTransformer(**cfg)
    model.init_weights()
    model.train()

    template = torch.randn(1, 3, 128, 128)
    online_template = torch.randn(1, 3, 128, 128)
    search = torch.randn(1, 3, 320, 320)
    template_feat, search_feat = model(template, online_template, search)
    assert template_feat.shape == torch.Size([1, 384, 8, 8])
    assert search_feat.shape == torch.Size([1, 384, 20, 20])
