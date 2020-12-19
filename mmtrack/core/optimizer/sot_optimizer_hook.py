import torch.nn as nn
from mmcv.runner.hooks import HOOKS, OptimizerHook


@HOOKS.register_module()
class SiameseRPNOptimizerHook(OptimizerHook):

    def __init__(self, backbone_start_train_epoch, backbone_train_layers,
                 **kwargs):
        super(SiameseRPNOptimizerHook, self).__init__(**kwargs)
        self.backbone_start_train_epoch = backbone_start_train_epoch
        self.backbone_train_layers = backbone_train_layers

    def before_train_epoch(self, runner):
        if runner.epoch >= self.backbone_start_train_epoch:
            for layer in self.backbone_train_layers:
                for param in getattr(runner.model.module.backbone,
                                     layer).parameters():
                    param.requires_grad = True
                for m in getattr(runner.model.module.backbone,
                                 layer).modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.train()
