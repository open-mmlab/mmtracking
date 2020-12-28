import torch.nn as nn
from mmcv.utils import Registry, build_from_cfg
from mmdet.models import DETECTORS

MODELS = Registry('model')
TRACKERS = Registry('tracker')
MOTION = Registry('motion')
REID = Registry('reid')
AGGREGATORS = Registry('aggregator')


def build(cfg, registry, default_args=None):
    """Build a module.

    Args:
        cfg (dict, list[dict]): The config of modules, is is either a dict
            or a list of configs.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Defaults to None.

    Returns:
        nn.Module: A built nn module.
    """
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        try:
            return nn.Sequential(*modules)
        except:  # noqa: E722
            return modules
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_tracker(cfg):
    """Build tracker."""
    return build(cfg, TRACKERS)


def build_motion(cfg):
    """Build motion model."""
    return build(cfg, MOTION)


def build_reid(cfg):
    """Build motion model."""
    return build(cfg, REID)


def build_aggregator(cfg):
    """Build aggregator model."""
    return build(cfg, AGGREGATORS)


def build_detector(cfg, train_cfg=None, test_cfg=None):
    """Build detector."""
    if train_cfg is None and test_cfg is None:
        return build(cfg, DETECTORS)
    else:
        return build(cfg, DETECTORS,
                     dict(train_cfg=train_cfg, test_cfg=test_cfg))


def build_model(cfg, train_cfg=None, test_cfg=None):
    """Build model."""
    if train_cfg is None and test_cfg is None:
        return build(cfg, MODELS)
    else:
        return build(cfg, MODELS, dict(train_cfg=train_cfg, test_cfg=test_cfg))
