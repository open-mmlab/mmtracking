# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry

MODELS = Registry('models', parent=MMCV_MODELS)
TRACKERS = MODELS
MOTION = MODELS
REID = MODELS
AGGREGATORS = MODELS


def build_tracker(cfg):
    """Build tracker."""
    return TRACKERS.build(cfg)


def build_motion(cfg):
    """Build motion model."""
    return MOTION.build(cfg)


def build_reid(cfg):
    """Build reid model."""
    return REID.build(cfg)


def build_aggregator(cfg):
    """Build aggregator model."""
    return AGGREGATORS.build(cfg)


def build_model(cfg, train_cfg=None, test_cfg=None):
    """Build model."""
    if train_cfg is None and test_cfg is None:
        return MODELS.build(cfg)
    else:
        return MODELS.build(cfg, MODELS,
                            dict(train_cfg=train_cfg, test_cfg=test_cfg))
