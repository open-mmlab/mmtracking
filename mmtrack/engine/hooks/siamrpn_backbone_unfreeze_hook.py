# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from torch.nn.modules.batchnorm import BatchNorm2d

from mmtrack.registry import HOOKS


@HOOKS.register_module()
class SiamRPNBackboneUnfreezeHook(Hook):
    """Start to train the backbone of SiamRPN++ from a certrain epoch.

    Args:
        backbone_start_train_epoch (int): Start to train the backbone at
            `backbone_start_train_epoch`-th epoch. Note the epoch in this
            class counts from 0, while the epoch in the log file counts from 1.
        backbone_train_layers (list(str)): List of str denoting the stages
            needed be trained in backbone.
    """

    def __init__(self,
                 backbone_start_train_epoch: int = 10,
                 backbone_train_layers: List = ['layer2', 'layer3', 'layer4']):
        self.backbone_start_train_epoch = backbone_start_train_epoch
        self.backbone_train_layers = backbone_train_layers

    def before_train_epoch(self, runner):
        """If `runner.epoch >= self.backbone_start_train_epoch`, start to train
        the backbone."""
        if runner.epoch >= self.backbone_start_train_epoch:
            for layer in self.backbone_train_layers:
                model = runner.model.module if is_model_wrapper(
                    runner.model) else runner.model
                for param in getattr(model.backbone, layer).parameters():
                    param.requires_grad = True
                for m in getattr(model.backbone, layer).modules():
                    if isinstance(m, BatchNorm2d):
                        m.train()
