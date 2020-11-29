import torch
import torch.nn as nn

from ..builder import AGGREGATORS


@AGGREGATORS.register_module()
class SelsaAggregator(nn.Module):
    """Selsa aggregator module.

    This module is proposed in
    "Sequence Level Semantics Aggregation for Video Object Detection".
    Link: https://arxiv.org/abs/1907.06390

    Args:
        in_channels (int): The number of channels of the features of proposal.
        num_attention_blocks (int): The number of attention blocks used in
            SELSA module. Default: 16.

    Attributes:
        fc_embed (nn.Linear): Fc layer used to embed the features of target
            proposals.
        ref_fc_embed (nn.Linear): Fc layer used to embed the features of
            support proposals.
        fc (nn.Linear): Fc layer used to transform the final features of
            target proposals.
        ref_fc (nn.Linear): Fc layer used to transform the features of support
            proposals.
        num_attention_blocks (int): The number of attention blocks used in
            SELSA module. Default: 16.
    """

    def __init__(self, in_channels, num_attention_blocks=16):
        super(SelsaAggregator, self).__init__()
        self.fc_embed = nn.Linear(in_channels, in_channels)
        self.ref_fc_embed = nn.Linear(in_channels, in_channels)
        self.fc = nn.Linear(in_channels, in_channels)
        self.ref_fc = nn.Linear(in_channels, in_channels)
        self.num_attention_blocks = num_attention_blocks

    def forward(self, x, ref_x):
        roi_n = x.shape[0]
        ref_roi_n = ref_x.shape[0]

        x_embed = self.fc_embed(x)
        # [num_attention_blocks, roi_n, C / num_attention_blocks]
        x_embed = x_embed.view(roi_n, self.num_attention_blocks,
                               -1).permute(1, 0, 2)

        ref_x_embed = self.ref_fc_embed(ref_x)
        # [num_attention_blocks, C / num_attention_blocks, ref_roi_n]
        ref_x_embed = ref_x_embed.view(ref_roi_n, self.num_attention_blocks,
                                       -1).permute(1, 2, 0)

        # [num_attention_blocks, roi_n, ref_roi_n]
        weights = torch.bmm(x_embed, ref_x_embed) / (x_embed.shape[-1]**0.5)
        weights = weights.softmax(dim=2)

        ref_x_new = self.ref_fc(ref_x)
        # [num_attention_blocks, ref_roi_n, C / num_attention_blocks]
        ref_x_new = ref_x_new.view(ref_roi_n, self.num_attention_blocks,
                                   -1).permute(1, 0, 2)

        # [roi_n, num_attention_blocks, C / num_attention_blocks]
        x_new = torch.bmm(weights, ref_x_new).permute(1, 0, 2).contiguous()
        # [roi_n, C]
        x_new = self.fc(x_new.view(roi_n, -1))
        return x_new
