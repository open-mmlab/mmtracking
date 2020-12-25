import torch
import torch.nn as nn
from mmcv.cnn.bricks import ConvModule

from ..builder import AGGREGATORS


@AGGREGATORS.register_module()
class EmbedAggregator(nn.Module):
    """Embed convs to aggregate several feature maps.

    This module is proposed in "Flow-Guided Feature Aggregation for Video
    Object Detection". `FGFA <https://arxiv.org/abs/1703.10025>`_.
    """

    def __init__(self,
                 num_convs=1,
                 channels=256,
                 kernel_size=3,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU')):
        """Initialization of EmbedAggregator.

        Args:
            num_convs (int): Number of embed convs.
            channels (int): Channels of embed convs. Defaults to 256.
            kernel_size (int): Kernel size of embed convs, Defaults to 3.
            norm_cfg (dict): Configuration of normlization method after each
                conv. Defaults to None.
            act_cfg (dict): Configuration of activation method after each
                conv. Defaults to dict(type='ReLU').
        """
        super(EmbedAggregator, self).__init__()
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

    def forward(self, x, ref_x):
        """Aggregate ref_x.

        The aggregation mainly contains two steps:
        1. Computing the cos similarity between x and ref_x.
        2. Use the normlized (i.e. softmax) cos similarity to weightedly sum
        ref_x.

        Args:
            x (Tensor): of shape [1, C, H, W]
            ref_x (Tensor): of shape [N, C, H, W]. N is the number of ref
                feature maps.

        Returns:
            Tensor: The aggregated feature map with shape [1, C, H, W].
        """
        assert len(x.shape) == 4 and len(x) == 1, \
            "Only support 'batch_size == 1' for x"
        x_embed = x
        for embed_conv in self.embed_convs:
            x_embed = embed_conv(x_embed)
        x_embed = x_embed / x_embed.norm(p=2, dim=1, keepdim=True)

        ref_x_embed = ref_x
        for embed_conv in self.embed_convs:
            ref_x_embed = embed_conv(ref_x_embed)
        ref_x_embed = ref_x_embed / ref_x_embed.norm(p=2, dim=1, keepdim=True)

        ada_weights = torch.sum(ref_x_embed * x_embed, dim=1, keepdim=True)
        ada_weights = ada_weights.softmax(dim=0)
        agg_x = torch.sum(ref_x * ada_weights, dim=0, keepdim=True)
        return agg_x
