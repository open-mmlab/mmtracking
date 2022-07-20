# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch.nn as nn
from mmdet.models import ConvFCBBoxHead
from torch import Tensor

from mmtrack.registry import MODELS
from mmtrack.utils import ConfigType


@MODELS.register_module()
class SelsaBBoxHead(ConvFCBBoxHead):
    """Selsa bbox head.

    This module is proposed in "Sequence Level Semantics Aggregation for Video
    Object Detection". `SELSA <https://arxiv.org/abs/1907.06390>`_.

    Args:
        aggregator (ConfigType): Configuration of aggregator.
    """

    def __init__(self, aggregator: ConfigType, *args, **kwargs):
        super(SelsaBBoxHead, self).__init__(*args, **kwargs)
        self.aggregator = nn.ModuleList()
        for i in range(self.num_shared_fcs):
            self.aggregator.append(MODELS.build(aggregator))
        self.inplace_false_relu = nn.ReLU(inplace=False)

    def forward(self, x: Tensor, ref_x: Tensor) -> Tuple:
        """Computing the `cls_score` and `bbox_pred` of the features `x` of key
        frame proposals.

        Args:
            x (Tensor): of shape [N, C, H, W]. N is the number of key frame
                proposals.
            ref_x (Tensor): of shape [M, C, H, W]. M is the number of reference
                frame proposals.

        Returns:
            tuple: A tuple of classification scores and bbox prediction.

                - cls_score (Tensor): Classification scores for all
                  scale levels, each is a 4D-tensor, the channels number
                  is num_base_priors * num_classes.
                - bbox_pred (Tensor): Box energies / deltas for all
                  scale levels, each is a 4D-tensor, the channels number
                  is num_base_priors * 4.
        """
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)
                ref_x = conv(ref_x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)
                ref_x = self.avg_pool(ref_x)

            x = x.flatten(1)
            ref_x = ref_x.flatten(1)

            for i, fc in enumerate(self.shared_fcs):
                x = fc(x)
                ref_x = fc(ref_x)
                x = x + self.aggregator[i](x, ref_x)
                ref_x = self.inplace_false_relu(ref_x)
                x = self.inplace_false_relu(x)

        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred
