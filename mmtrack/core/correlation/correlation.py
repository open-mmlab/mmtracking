import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks import ConvModule


class CorrelationHead(nn.Module):

    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 kernel_size=3,
                 norm_cfg=dict(type='BN')):
        super(CorrelationHead, self).__init__()
        self.kernel_convs = ConvModule(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            norm_cfg=norm_cfg)

        self.search_convs = ConvModule(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            norm_cfg=norm_cfg)

        self.head_convs = nn.Sequential(
            ConvModule(
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=1,
                norm_cfg=norm_cfg),
            ConvModule(
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_size=1,
                act_cfg=None))

    def forward(self, kernel, search):
        kernel = self.kernel_convs(kernel)
        search = self.search_convs(search)
        correlation_maps = depthwise_correlation(search, kernel)
        out = self.head_convs(correlation_maps)
        return out


def depthwise_correlation(x, kernel):
    """depthwise cross correlation."""
    batch = kernel.size(0)
    channel = kernel.size(1)
    x = x.view(1, batch * channel, x.size(2), x.size(3))
    kernel = kernel.view(batch * channel, 1, kernel.size(2), kernel.size(3))
    out = F.conv2d(x, kernel, groups=batch * channel)
    out = out.view(batch, channel, out.size(2), out.size(3))
    return out
