import torch.nn as nn
from mmcv.cnn import (build_activation_layer, build_norm_layer, constant_init,
                      kaiming_init)


class FcModule(nn.Module):
    """Fully-connected layer module.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Ourput channels.
        norm_cfg (dict, optional): Configuration of normlization method
            after fc. Defaults to None.
        act_cfg (dict, optional): Configuration of activation method after fc.
            Defaults to dict(type='ReLU').
        inplace (bool, optional): Whether inplace the activatation module.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 inplace=True):
        super(FcModule, self).__init__()
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert act_cfg is None or isinstance(act_cfg, dict)
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.inplace = inplace

        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None

        self.fc = nn.Linear(in_channels, out_channels)
        # build normalization layers
        if self.with_norm:
            self.norm_name, norm = build_norm_layer(norm_cfg, out_channels)
            self.add_module(self.norm_name, norm)

        # build activation layer
        if self.with_activation:
            act_cfg_ = act_cfg.copy()
            # nn.Tanh has no 'inplace' argument
            if act_cfg_['type'] not in [
                    'Tanh', 'PReLU', 'Sigmoid', 'HSigmoid', 'Swish'
            ]:
                act_cfg_.setdefault('inplace', inplace)
            self.activate = build_activation_layer(act_cfg_)

        # Use msra init by default
        self.init_weights()

    @property
    def norm(self):
        """Normalization."""
        return getattr(self, self.norm_name)

    def init_weights(self):
        """Initialize weights."""
        kaiming_init(self.fc)
        if self.with_norm:
            constant_init(self.norm, 1, bias=0)

    def forward(self, x, activate=True, norm=True):
        """Model forward."""
        x = self.fc(x)
        if norm and self.with_norm:
            x = self.norm(x)
        if activate and self.with_activation:
            x = self.activate(x)
        return x
