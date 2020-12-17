import torch.nn as nn
from mmcv.runner.hooks import HOOKS, OptimizerHook


@HOOKS.register_module()
class SiameseRPNOptimizerHook(OptimizerHook):

    def __init__(self, start_train_backbone_epoch, backbone_train_layers,
                 **kwargs):
        super(SiameseRPNOptimizerHook, self).__init__(**kwargs)
        self.start_train_backbone_epoch = start_train_backbone_epoch
        self.backbone_train_layers = backbone_train_layers

    def before_train_epoch(self, runner):
        if runner.epoch >= self.start_train_backbone_epoch:
            for layer in self.backbone_train_layers:
                for param in getattr(runner.model.module.backbone,
                                     layer).parameters():
                    param.requires_grad = True
                for m in getattr(runner.model.module.backbone,
                                 layer).modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.train()
