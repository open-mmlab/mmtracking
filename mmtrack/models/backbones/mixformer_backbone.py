# Copyright (c) OpenMMLab. All rights reserved.
import logging
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.drop import DropPath
from mmcv.cnn.bricks.transformer import FFN
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.runner import BaseModule
from mmdet.models.builder import BACKBONES

from mmtrack.core.utils.misc import ntuple

to_2tuple = ntuple(2)


class LayerNormAutofp32(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    """Approximation of GELU activation function introduced in `Gaussian Error
    Linear Units<https://arxiv.org/abs/1606.08415v4>`."""

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class MixedAttentionModule(nn.Module):
    """Mixed Attention Module (MAM) proposed in MixFormer.

    It is the core design for simultaneous feature extraction
    and target information integration. Please refer to
    `MixFormer<https://arxiv.org/abs/2203.11082>`_ for more details.

    Args:
        dim_in (int): Input dimension of this module.
        dim_out (int): Output dimension of this module.
        num_heads (int): Number of heads in multi-head attention mechanism.
        qkv_bias (bool): Add bias when projecting to qkv tokens.
            Default: False
        attn_drop (float): A Dropout layer on attn_output_weight.
            Default: 0.0
        proj_drop (float): A Dropout layer after attention operation.
            Default: 0.0
        method (str): Method for building the projection layer. Choices are
            ['dw_bn', 'avg', 'identity']. Default: 'dw_bn'
        kernel_size (int): Kernel size of the projection layer. Default: 1
        stride_q (int): Stride of the query projection layer. Default: 1
        stride_kv (int): Stride of the key/value projection layer. Default: 1
        padding_q (int): Padding number of the query projection layer.
            Default: 1
        padding_kv (int): Padding number of the key/value projection layer.
            Default: 1
        norm_cfg (dict): Norm layer config.
    """

    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 method='dw_bn',
                 kernel_size=3,
                 stride_kv=1,
                 stride_q=1,
                 padding_kv=1,
                 padding_q=1,
                 norm_cfg=dict(type='BN'),
                 **kwargs):
        super().__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.dim = dim_out
        self.num_heads = num_heads
        self.scale = dim_out**-0.5
        self.norm_cfg = norm_cfg

        self.conv_proj_q = self._build_projection(
            dim_in, dim_out, kernel_size, padding_q, stride_q,
            'identity' if method == 'avg' else method)
        self.conv_proj_k = self._build_projection(dim_in, dim_out, kernel_size,
                                                  padding_kv, stride_kv,
                                                  method)
        self.conv_proj_v = self._build_projection(dim_in, dim_out, kernel_size,
                                                  padding_kv, stride_kv,
                                                  method)

        self.proj_q = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_k = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_v = nn.Linear(dim_in, dim_out, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_out, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

    def _build_projection(self, dim_in, dim_out, kernel_size, padding, stride,
                          method):
        """Build qkv projection according to `method` argument.

        Args:
            dim_in (int): input dimension
            dim_out (int): output dimension
            kernel_size (int): kernel size of convolution
            padding (int): padding of convolution
            stride (int): stride of convolution
            method (str): description of projection method.
                ``'dw_bn'``: Apply a convolution layer with batch
                                    norm on the input.
                ``'avg'``: Apply an avgpool2d on the input.
                ``'identity'``: No transformation on the input.
        """
        if method == 'dw_bn':
            proj = nn.Sequential(
                OrderedDict([
                    ('conv',
                     nn.Conv2d(
                         dim_in,
                         dim_in,
                         kernel_size=kernel_size,
                         padding=padding,
                         stride=stride,
                         bias=False,
                         groups=dim_in)),
                    build_norm_layer(self.norm_cfg, dim_in),
                ]))
        elif method == 'avg':
            proj = nn.Sequential(
                OrderedDict([
                    ('avg',
                     nn.AvgPool2d(
                         kernel_size=kernel_size,
                         padding=padding,
                         stride=stride,
                         ceil_mode=True)),
                ]))
        elif method == 'identity':
            proj = None
        else:
            raise ValueError('Unknown method ({})'.format(method))

        return proj

    def forward_conv(self, x, t_h, t_w, s_h, s_w):
        """Projecting the input to qkv tokens."""
        template, online_template, search = torch.split(
            x, [t_h * t_w, t_h * t_w, s_h * s_w], dim=1)
        template = rearrange(
            template, 'b (h w) c -> b c h w', h=t_h, w=t_w).contiguous()
        online_template = rearrange(
            online_template, 'b (h w) c -> b c h w', h=t_h,
            w=t_w).contiguous()
        search = rearrange(
            search, 'b (h w) c -> b c h w', h=s_h, w=s_w).contiguous()

        if self.conv_proj_q is not None:
            t_q = self.conv_proj_q(template)
            ot_q = self.conv_proj_q(online_template)
            s_q = self.conv_proj_q(search)
        else:
            t_q = template
            ot_q = online_template
            s_q = search

        t_q = rearrange(t_q, 'b c h w -> b (h w) c').contiguous()
        ot_q = rearrange(ot_q, 'b c h w -> b (h w) c').contiguous()
        s_q = rearrange(s_q, 'b c h w -> b (h w) c').contiguous()
        q = torch.cat([t_q, ot_q, s_q], dim=1)

        if self.conv_proj_k is not None:
            t_k = self.conv_proj_k(template)
            ot_k = self.conv_proj_k(online_template)
            s_k = self.conv_proj_k(search)
        else:
            t_k = template
            ot_k = online_template
            s_k = search

        t_k = rearrange(t_k, 'b c h w -> b (h w) c').contiguous()
        ot_k = rearrange(ot_k, 'b c h w -> b (h w) c').contiguous()
        s_k = rearrange(s_k, 'b c h w -> b (h w) c').contiguous()
        k = torch.cat([t_k, ot_k, s_k], dim=1)

        if self.conv_proj_v is not None:
            t_v = self.conv_proj_v(template)
            ot_v = self.conv_proj_v(online_template)
            s_v = self.conv_proj_v(search)
        else:
            t_v = template
            ot_v = online_template
            s_v = search

        t_v = rearrange(t_v, 'b c h w -> b (h w) c').contiguous()
        ot_v = rearrange(ot_v, 'b c h w -> b (h w) c').contiguous()
        s_v = rearrange(s_v, 'b c h w -> b (h w) c').contiguous()
        v = torch.cat([t_v, ot_v, s_v], dim=1)

        return q, k, v

    def forward_conv_test(self, x, s_h, s_w):
        search = rearrange(
            x, 'b (h w) c -> b c h w', h=s_h, w=s_w).contiguous()

        if self.conv_proj_q is not None:
            q = self.conv_proj_q(search)
        else:
            q = search
        q = rearrange(q, 'b c h w -> b (h w) c').contiguous()

        if self.conv_proj_k is not None:
            k = self.conv_proj_k(search)
        else:
            k = search
        k = rearrange(k, 'b c h w -> b (h w) c').contiguous()
        k = torch.cat([self.t_k, self.ot_k, k], dim=1)

        if self.conv_proj_v is not None:
            v = self.conv_proj_v(search)
        else:
            v = search
        v = rearrange(v, 'b c h w -> b (h w) c').contiguous()
        v = torch.cat([self.t_v, self.ot_v, v], dim=1)

        return q, k, v

    def forward(self, x, t_h, t_w, s_h, s_w):
        """Asymmetric mixed attention.

        Args:
            x (Tensor): concatenated feature of temmlate and search,
                shape (B, 2*t_h*t_w+s_h*s_w, C)
            t_h (int): template feature height
            t_w (int): template feature width
            s_h (int): search region feature height
            s_w (int): search region feature width
        """
        if (self.conv_proj_q is not None or self.conv_proj_k is not None
                or self.conv_proj_v is not None):
            q, k, v = self.forward_conv(x, t_h, t_w, s_h, s_w)

        q = rearrange(
            self.proj_q(q), 'b t (h d) -> b h t d',
            h=self.num_heads).contiguous()
        k = rearrange(
            self.proj_k(k), 'b t (h d) -> b h t d',
            h=self.num_heads).contiguous()
        v = rearrange(
            self.proj_v(v), 'b t (h d) -> b h t d',
            h=self.num_heads).contiguous()

        # Attention!: k/v compression，1/4 of q_size（conv_stride=2）

        q_mt, q_s = torch.split(q, [t_h * t_w * 2, s_h * s_w], dim=2)
        k_mt, k_s = torch.split(
            k, [((t_h + 1) // 2)**2 * 2, s_h * s_w // 4], dim=2)
        v_mt, v_s = torch.split(
            v, [((t_h + 1) // 2)**2 * 2, s_h * s_w // 4], dim=2)

        # template attention
        attn_score = torch.einsum('bhlk,bhtk->bhlt', [q_mt, k_mt]) * self.scale
        attn = F.softmax(attn_score, dim=-1)
        attn = self.attn_drop(attn)
        x_mt = torch.einsum('bhlt,bhtv->bhlv', [attn, v_mt])
        x_mt = rearrange(x_mt, 'b h t d -> b t (h d)')

        # search region attention
        attn_score = torch.einsum('bhlk,bhtk->bhlt', [q_s, k]) * self.scale
        attn = F.softmax(attn_score, dim=-1)
        attn = self.attn_drop(attn)
        x_s = torch.einsum('bhlt,bhtv->bhlv', [attn, v])
        x_s = rearrange(x_s, 'b h t d -> b t (h d)')

        x = torch.cat([x_mt, x_s], dim=1)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def forward_test(self, x, s_h, s_w):
        if (self.conv_proj_q is not None or self.conv_proj_k is not None
                or self.conv_proj_v is not None):
            q_s, k, v = self.forward_conv_test(x, s_h, s_w)

        q_s = rearrange(
            self.proj_q(q_s), 'b t (h d) -> b h t d',
            h=self.num_heads).contiguous()
        k = rearrange(
            self.proj_k(k), 'b t (h d) -> b h t d',
            h=self.num_heads).contiguous()
        v = rearrange(
            self.proj_v(v), 'b t (h d) -> b h t d',
            h=self.num_heads).contiguous()

        attn_score = torch.einsum('bhlk,bhtk->bhlt', [q_s, k]) * self.scale
        attn = F.softmax(attn_score, dim=-1)
        attn = self.attn_drop(attn)
        x_s = torch.einsum('bhlt,bhtv->bhlv', [attn, v])
        x_s = rearrange(x_s, 'b h t d -> b t (h d)').contiguous()

        x = x_s

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def set_online(self, x, t_h, t_w):
        template = x[:, :t_h * t_w]  # 1, 1024, c
        online_template = x[:, t_h * t_w:]  # 1, b*1024, c
        template = rearrange(
            template, 'b (h w) c -> b c h w', h=t_h, w=t_w).contiguous()
        online_template = rearrange(
            online_template.squeeze(0), '(b h w) c -> b c h w', h=t_h,
            w=t_w).contiguous()  # b, c, 32, 32

        if self.conv_proj_q is not None:
            t_q = rearrange(
                self.conv_proj_q(template),
                'b c h w -> b (h w) c').contiguous()
            ot_q = rearrange(
                self.conv_proj_q(online_template),
                'b c h w -> (b h w) c').contiguous().unsqueeze(0)
        else:
            t_q = rearrange(template, 'b c h w -> b (h w) c').contiguous()
            ot_q = rearrange(online_template,
                             'b c h w -> (b h w) c').contiguous().unsqueeze(0)
        q = torch.cat([t_q, ot_q], dim=1)

        if self.conv_proj_k is not None:
            self.t_k = rearrange(
                self.conv_proj_k(template),
                'b c h w -> b (h w) c').contiguous()
            self.ot_k = rearrange(
                self.conv_proj_k(online_template),
                'b c h w -> (b h w) c').contiguous().unsqueeze(0)
        else:
            self.t_k = rearrange(template, 'b c h w -> b (h w) c').contiguous()
            self.ot_k = rearrange(
                online_template,
                'b c h w -> (b h w) c').contiguous().unsqueeze(0)
        k = torch.cat([self.t_k, self.ot_k], dim=1)

        if self.conv_proj_v is not None:
            self.t_v = rearrange(
                self.conv_proj_v(template),
                'b c h w -> b (h w) c').contiguous()
            self.ot_v = rearrange(
                self.conv_proj_v(online_template),
                'b c h w -> (b h w) c').contiguous().unsqueeze(0)
        else:
            self.t_v = rearrange(template, 'b c h w -> b (h w) c').contiguous()
            self.ot_v = rearrange(
                online_template,
                'b c h w -> (b h w) c').contiguous().unsqueeze(0)
        v = torch.cat([self.t_v, self.ot_v], dim=1)

        q = rearrange(
            self.proj_q(q), 'b t (h d) -> b h t d',
            h=self.num_heads).contiguous()
        k = rearrange(
            self.proj_k(k), 'b t (h d) -> b h t d',
            h=self.num_heads).contiguous()
        v = rearrange(
            self.proj_v(v), 'b t (h d) -> b h t d',
            h=self.num_heads).contiguous()

        attn_score = torch.einsum('bhlk,bhtk->bhlt', [q, k]) * self.scale
        attn = F.softmax(attn_score, dim=-1)
        attn = self.attn_drop(attn)
        x = torch.einsum('bhlt,bhtv->bhlv', [attn, v])
        x = rearrange(x, 'b h t d -> b t (h d)').contiguous()

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MixFormerAttentionBlock(nn.Module):
    """Block containing attention operation, FFN and residual layer."""

    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 mlp_channel_ratio=4.,
                 qkv_bias=False,
                 drop=0.,
                 attn_drop=0.,
                 path_drop_probs=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 norm_cfg=dict(type='BN'),
                 **kwargs):
        super().__init__()

        self.norm1 = norm_layer(dim_in)
        self.attn = MixedAttentionModule(
            dim_in,
            dim_out,
            num_heads,
            qkv_bias,
            attn_drop,
            drop,
            norm_cfg=norm_cfg,
            **kwargs)

        self.drop_path = DropPath(path_drop_probs) \
            if path_drop_probs > 0. else nn.Identity()
        self.norm2 = norm_layer(dim_out)

        dim_mlp_hidden = int(dim_out * mlp_channel_ratio)
        self.mlp = FFN(
            embed_dims=dim_out,
            feedforward_channels=dim_mlp_hidden,
            num_fcs=2,
            act_cfg=dict(type='GELU'),
            ffn_drop=drop,
            add_identity=False,
        )

    def forward(self, x, t_h, t_w, s_h, s_w):
        """
        Args:
            x (Tensor): concatenated feature of temmlate and search,
                shape (B, 2*t_h*t_w+s_h*s_w, C)
            t_h (int): template feature height
            t_w (int): template feature width
            s_h (int): search region feature height
            s_w (int): search region feature width
        """
        res = x
        x = self.norm1(x)
        attn = self.attn(x, t_h, t_w, s_h, s_w)
        x = res + self.drop_path(attn)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def forward_test(self, x, s_h, s_w):
        res = x

        x = self.norm1(x)
        attn = self.attn.forward_test(x, s_h, s_w)
        x = res + self.drop_path(attn)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def set_online(self, x, t_h, t_w):
        res = x
        x = self.norm1(x)
        attn = self.attn.set_online(x, t_h, t_w)
        x = res + self.drop_path(attn)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ConvEmbed(nn.Module):
    """Image to Conv Embedding.

    Args:
        patch_size (int): patch size
        in_chans (int): number of input channels
        embed_dim (int): embedding dimension
        stride (int): stride of convolution layer
        padding (int): number of padding
        norm_layer (nn.Module): normalization layer
    """

    def __init__(self,
                 patch_size=7,
                 in_chans=3,
                 embed_dim=64,
                 stride=4,
                 padding=2,
                 norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        x = self.proj(x)

        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.norm:
            x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W).contiguous()

        return x


class ConvVisionTransformerLayer(BaseModule):
    """One stage of ConvVisionTransformer containing one patch embed layer and
    stacked attention blocks.

    Args:
        patch_size (int): patch size of ConvEmbed module
        patch_stride (int): patch stride of ConvEmbed module
        patch_padding (int): padding of ConvEmbed module
        in_chans (int): number of input channels
        embed_dim (int): embedding dimension
        depth (int): number of attention blocks
        num_heads (int): number of heads in multi-head attention operation
        mlp_channel_ratio (int): hidden dim ratio of FFN
        qkv_bias (bool): qkv bias
        drop_rate (float): drop rate after patch embed
        attn_drop_rate (float): drop rate in attention
        path_drop_probs (float): drop path for stochastic depth decay
        act_layer (nn.Module): activate function used in FFN
        norm_layer (nn.Module): normalization layer used in attention block
        init (str): weight init method
        norm_cfg (dict): normalization layer config
    """

    def __init__(self,
                 patch_size=16,
                 patch_stride=16,
                 patch_padding=0,
                 in_chans=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_channel_ratio=4.,
                 qkv_bias=False,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 path_drop_probs=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 init='trunc_norm',
                 norm_cfg=False,
                 **kwargs):
        super().__init__()
        self.init = init
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = ConvEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            stride=patch_stride,
            padding=patch_padding,
            embed_dim=embed_dim,
            norm_layer=norm_layer)

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, path_drop_probs, depth)
               ]  # stochastic depth decay rule

        blocks = []
        for j in range(depth):
            blocks.append(
                MixFormerAttentionBlock(
                    dim_in=embed_dim,
                    dim_out=embed_dim,
                    num_heads=num_heads,
                    mlp_channel_ratio=mlp_channel_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    path_drop_probs=dpr[j],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    norm_cfg=norm_cfg,
                    **kwargs))
        self.blocks = nn.ModuleList(blocks)

    def init_weights(self):
        if self.init == 'xavier':
            self.apply(self._init_weights_xavier)
        else:
            self.apply(self._init_weights_trunc_normal)

    def _init_weights_trunc_normal(self, m):
        if isinstance(m, nn.Linear):
            logging.info('=> init weight of Linear from trunc norm')
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                logging.info('=> init bias of Linear to zeros')
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_weights_xavier(self, m):
        if isinstance(m, nn.Linear):
            logging.info('=> init weight of Linear from xavier uniform')
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                logging.info('=> init bias of Linear to zeros')
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, template, online_template, search):
        """
        Args:
            template (Tensor): template features of shape (B, C, H, W)
            online template (Tensor): online template features
                of shape (B, C, H, W)
            search (Tensor): search features of shape (B, C, H, W)
        """
        template = self.patch_embed(template)
        online_template = self.patch_embed(online_template)
        t_B, t_C, t_H, t_W = template.size()
        search = self.patch_embed(search)
        s_B, s_C, s_H, s_W = search.size()

        template = rearrange(template, 'b c h w -> b (h w) c').contiguous()
        online_template = rearrange(online_template,
                                    'b c h w -> b (h w) c').contiguous()
        search = rearrange(search, 'b c h w -> b (h w) c').contiguous()
        x = torch.cat([template, online_template, search], dim=1)

        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x, t_H, t_W, s_H, s_W)

        template, online_template, search = torch.split(
            x, [t_H * t_W, t_H * t_W, s_H * s_W], dim=1)
        template = rearrange(
            template, 'b (h w) c -> b c h w', h=t_H, w=t_W).contiguous()
        online_template = rearrange(
            online_template, 'b (h w) c -> b c h w', h=t_H,
            w=t_W).contiguous()
        search = rearrange(
            search, 'b (h w) c -> b c h w', h=s_H, w=s_W).contiguous()

        return template, online_template, search

    def forward_test(self, search):
        search = self.patch_embed(search)
        s_B, s_C, s_H, s_W = search.size()

        search = rearrange(search, 'b c h w -> b (h w) c').contiguous()

        x = self.pos_drop(search)

        for i, blk in enumerate(self.blocks):
            x = blk.forward_test(x, s_H, s_W)

        search = rearrange(x, 'b (h w) c -> b c h w', h=s_H, w=s_W)

        return search

    def set_online(self, template, online_template):
        template = self.patch_embed(template)
        online_template = self.patch_embed(online_template)
        t_B, t_C, t_H, t_W = template.size()

        template = rearrange(template, 'b c h w -> b (h w) c').contiguous()
        online_template = rearrange(
            online_template, 'b c h w -> (b h w) c').unsqueeze(0).contiguous()
        x = torch.cat([template, online_template], dim=1)

        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk.set_online(x, t_H, t_W)

        template = x[:, :t_H * t_W]
        online_template = x[:, t_H * t_W:]
        template = rearrange(template, 'b (h w) c -> b c h w', h=t_H, w=t_W)
        online_template = rearrange(
            online_template.squeeze(0), '(b h w) c -> b c h w', h=t_H, w=t_W)

        return template, online_template


@BACKBONES.register_module()
class ConvVisionTransformer(BaseModule):
    """Vision Transformer with support for patch or hybrid CNN input stage.

    This backbone refers to the implementation of
    `CvT: <https://arxiv.org/abs/2103.15808>`_.

    Args:
        in_chans (int): number of input channels
        act_layer (nn.Module): activate function used in FFN
        norm_layer (nn.Module): normalization layer used in attention block
        init (str): weight init method
        num_stage (int): number of backbone stages
        patch_size (List[int]): patch size of each stage
        patch_stride (List[int]): patch stride of each stage
        patch_padding (List[int]): patch padding of each stage
        dim_embed (List[int]): embedding dimension of each stage
        num_heads (List[int]): number of heads in multi-head
        attention operation of each stage
        depth (List[int]): number of attention blocks of each stage
        mlp_channel_ratio (List[int]): hidden dim ratio of FFN of each stage
        attn_drop_rate (List[float]): attn drop rate of each stage
        drop_rate (List[float]): drop rate of each stage
        path_drop_probs (List[float]): drop path of each stage
        qkv_bias (List[bool]): qkv bias of each stage
        qkv_proj_method (List[str]): qkv project method of each stage
        kernel_qkv (List[int]): kernel size for qkv projection of each stage
        padding_kv/q (List[int]): padding size for kv/q projection
        of each stage
        stride_kv/q (List[int]): stride for kv/q project of each stage
        norm_cfg (dict): normalization layer config
    """

    def __init__(self,
                 in_chans=3,
                 act_layer=QuickGELU,
                 norm_layer=partial(LayerNormAutofp32, eps=1e-5),
                 init='trunc_norm',
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
                 norm_cfg=dict(type='BN', requires_grad=False)):
        super().__init__()

        self.num_stages = num_stages
        for i in range(self.num_stages):
            kwargs = {
                'patch_size': patch_size[i],
                'patch_stride': patch_stride[i],
                'patch_padding': patch_padding[i],
                'embed_dim': dim_embed[i],
                'depth': depth[i],
                'num_heads': num_heads[i],
                'mlp_channel_ratio': mlp_channel_ratio[i],
                'qkv_bias': qkv_bias[i],
                'drop_rate': drop_rate[i],
                'attn_drop_rate': attn_drop_rate[i],
                'path_drop_probs': path_drop_probs[i],
                'method': qkv_proj_method[i],
                'kernel_size': kernel_qkv[i],
                'padding_q': padding_q[i],
                'padding_kv': padding_kv[i],
                'stride_kv': stride_kv[i],
                'stride_q': stride_q[i],
                'norm_cfg': norm_cfg,
            }

            stage = ConvVisionTransformerLayer(
                in_chans=in_chans,
                init=init,
                act_layer=act_layer,
                norm_layer=norm_layer,
                **kwargs)
            setattr(self, f'stage{i}', stage)

            in_chans = dim_embed[i]

        dim_embed = dim_embed[-1]
        self.norm = norm_layer(dim_embed)

        self.head = nn.Linear(dim_embed, 1000)

    def forward(self, template, online_template, search):
        """Forward-pass method in train pipeline.

        Args:
            template (Tensor): template images of shape (B, C, H, W)
            online template (Tensor): online template images
            of shape (B, C, H, W)
            search (Tensor): search images of shape (B, C, H, W)
        """
        for i in range(self.num_stages):
            template, online_template, search = getattr(self, f'stage{i}')(
                template, online_template, search)

        return template, search

    def forward_test(self, search):
        """Forward-pass method for search image in test pipeline. The model
        forwarding strategies are different between train and test. In test
        pipeline, we call ``search()`` method which only takes in search image
        when tracker is tracking current frame. This approach reduces
        computational overhead and thus increases tracking speed.

        Args:
            search (Tensor): search images of shape (B, C, H, W)
        """
        for i in range(self.num_stages):
            search = getattr(self, f'stage{i}').forward_test(search)
        return self.template, search

    def set_online(self, template, online_template):
        """Forward-pass method for template image in test pipeline. The model
        forwarding strategies are different between train and test. In test
        pipeline, we call ``set_online()`` method which only takes in template
        images when tracker is initialized or is updating online template. This
        approach reduces computational overhead and thus increases tracking
        speed.

        Args:
            template (Tensor): template images of shape (B, C, H, W)
            online template (Tensor): online template images
            of shape (B, C, H, W)
        """
        for i in range(self.num_stages):
            template, online_template = getattr(self, f'stage{i}').set_online(
                template, online_template)
        self.template = template
