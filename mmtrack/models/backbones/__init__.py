from mmdet.models.backbones import ResNet, ResNeXt

from ..builder import BACKBONES

__all__ = ['ResNet', 'ResNeXt']

BACKBONES.register_module(ResNet)
BACKBONES.register_module(ResNeXt)
