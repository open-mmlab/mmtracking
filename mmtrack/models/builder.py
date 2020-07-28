import torch.nn as nn
from mmcv.utils import Registry, build_from_cfg

MODELS = Registry('model')


def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_model(cfg, train_cfg=None, test_cfg=None):
    return build(cfg, MODELS, dict(train_cfg=train_cfg, test_cfg=test_cfg))
