import warnings

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
    """Build motion model."""
    return REID.build(cfg)


def build_aggregator(cfg):
    """Build aggregator model."""
    return AGGREGATORS.build(cfg)


def build_model(cfg, train_cfg=None, test_cfg=None):
    """Build model."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '
    return MODELS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))
