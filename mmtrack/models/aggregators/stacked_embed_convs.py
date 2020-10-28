import torch
import torch.nn as nn
from mmcv.cnn.bricks import ConvModule

from ..builder import AGGREGATORS


@AGGREGATORS.register_module()
class StackedEmbedConvs(nn.Module):

    def __init__(self,
                 num_convs=1,
                 channels=256,
                 kernel_size=3,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU')):
        super(StackedEmbedConvs, self).__init__()
        assert num_convs > 0, 'The number of convs must be bigger than 1.'
        self.embed_convs = nn.ModuleList()
        for i in range(num_convs):
            if i == num_convs - 1:
                new_norm_cfg = None
                new_act_cfg = None
            else:
                new_norm_cfg = norm_cfg
                new_act_cfg = act_cfg
            self.embed_convs.append(
                ConvModule(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) // 2,
                    norm_cfg=new_norm_cfg,
                    act_cfg=new_act_cfg))

    def forward(self, target_x, ref_x):
        assert len(target_x.shape) == 4 and len(target_x) == 1, \
            "Only support 'batch_size == 1' for target_x"
        target_x_embed = target_x
        for embed_conv in self.embed_convs:
            target_x_embed = embed_conv(target_x_embed)
        target_x_embed = target_x_embed / target_x_embed.norm(
            p=2, dim=1, keepdim=True)

        ref_x_embed = ref_x
        for embed_conv in self.embed_convs:
            ref_x_embed = embed_conv(ref_x_embed)
        ref_x_embed = ref_x_embed / ref_x_embed.norm(p=2, dim=1, keepdim=True)

        ada_weights = torch.sum(
            ref_x_embed * target_x_embed, dim=1, keepdim=True)
        ada_weights = ada_weights.softmax(dim=0)
        agg_x = torch.sum(ref_x * ada_weights, dim=0, keepdim=True)
        return agg_x
