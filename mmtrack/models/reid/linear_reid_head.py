import torch.nn as nn
from mmcls.models import ClsHead
from mmcls.models.builder import HEADS, build_loss
from mmcv.cnn import normal_init

from .fc_module import FcModule


@HEADS.register_module()
class LinearReIDHead(ClsHead):

    def __init__(self,
                 num_fcs,
                 in_channels,
                 fc_channels,
                 out_channels,
                 norm_cfg=None,
                 act_cfg=None,
                 num_classes=None,
                 loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 topk=(1, )):
        super(LinearReIDHead, self).__init__(loss=loss, topk=topk)
        self.num_fcs = num_fcs
        self.in_channels = in_channels
        self.fc_channels = fc_channels
        self.out_channels = out_channels
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.num_classes = num_classes
        self.loss = build_loss(loss)

        self._init_layers()

    def _init_layers(self):
        self.fcs = nn.ModuleList()
        for i in range(self.num_fcs):
            in_channels = self.in_channels if i == 0 else self.fc_channels
            self.fcs.append(
                FcModule(in_channels, self.fc_channels, self.norm_cfg,
                         self.act_cfg))
        in_channels = self.in_channels if self.num_fcs == 0 else \
            self.fc_channels
        self.fc_out = nn.Linear(in_channels, self.out_channels)

    def init_weights(self):
        normal_init(self.fc_out, mean=0, std=0.01, bias=0)

    def simple_test(self, x):
        """Test without augmentation."""
        for m in self.fcs:
            x = m(x)
        x = self.fc_out(x)
        return x

    def forward_train(self, x, gt_label):
        x = self.fcs(x)
        x = self.fc_out(x)
        if self.num_classes is not None:
            raise NotImplementedError()
        losses = self.loss(x, gt_label)
        return losses
