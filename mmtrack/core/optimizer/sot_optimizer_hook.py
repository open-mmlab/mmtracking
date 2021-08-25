# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.runner.hooks import HOOKS, Fp16OptimizerHook, OptimizerHook


@HOOKS.register_module()
class SiameseRPNOptimizerHook(OptimizerHook):
    """Optimizer hook for siamese rpn.

    Args:
        backbone_start_train_epoch (int): Start to train the backbone at
            `backbone_start_train_epoch`-th epoch. Note the epoch in this
            class counts from 0, while the epoch in the log file counts from 1.
        backbone_train_layers (list(str)): List of str denoting the stages
            needed be trained in backbone.
    """

    def __init__(self, backbone_start_train_epoch, backbone_train_layers,
                 **kwargs):
        super(SiameseRPNOptimizerHook, self).__init__(**kwargs)
        self.backbone_start_train_epoch = backbone_start_train_epoch
        self.backbone_train_layers = backbone_train_layers

    def before_train_epoch(self, runner):
        """If `runner.epoch >= self.backbone_start_train_epoch`, start to train
        the backbone."""
        if runner.epoch >= self.backbone_start_train_epoch:
            for layer in self.backbone_train_layers:
                for param in getattr(runner.model.module.backbone,
                                     layer).parameters():
                    param.requires_grad = True
                for m in getattr(runner.model.module.backbone,
                                 layer).modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.train()


@HOOKS.register_module()
class SiameseRPNFp16OptimizerHook(Fp16OptimizerHook):
    """FP16Optimizer hook for siamese rpn.

    Args:
        backbone_start_train_epoch (int): Start to train the backbone at
            `backbone_start_train_epoch`-th epoch. Note the epoch in this
            class counts from 0, while the epoch in the log file counts from 1.
        backbone_train_layers (list(str)): List of str denoting the stages
            needed be trained in backbone.
    """

    def __init__(self, backbone_start_train_epoch, backbone_train_layers,
                 **kwargs):
        super(SiameseRPNFp16OptimizerHook, self).__init__(**kwargs)
        self.backbone_start_train_epoch = backbone_start_train_epoch
        self.backbone_train_layers = backbone_train_layers

    def before_train_epoch(self, runner):
        """If `runner.epoch >= self.backbone_start_train_epoch`, start to train
        the backbone."""
        if runner.epoch >= self.backbone_start_train_epoch:
            for layer in self.backbone_train_layers:
                for param in getattr(runner.model.module.backbone,
                                     layer).parameters():
                    param.requires_grad = True
                for m in getattr(runner.model.module.backbone,
                                 layer).modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.train()
