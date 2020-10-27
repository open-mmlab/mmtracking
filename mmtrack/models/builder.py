from mmcv.utils import Registry
from mmdet.models import DETECTORS
from mmdet.models.builder import build

MODELS = Registry('model')
TRACKERS = Registry('tracker')
MOTION = Registry('motion')
AGGREGATORS = Registry('aggregator')


def build_tracker(cfg):
    """Build tracker."""
    return build(cfg, TRACKERS)


def build_motion(cfg):
    """Build motion model."""
    return build(cfg, MOTION)


def build_aggregator(cfg):
    """Build aggregator model."""
    return build(cfg, AGGREGATORS)


def build_detector(cfg):
    """Build detector."""
    return build(cfg, DETECTORS)


def build_model(cfg, train_cfg=None, test_cfg=None):
    """Build model."""
    if train_cfg is None and test_cfg is None:
        return build(cfg, MODELS)
    else:
        return build(cfg, MODELS, dict(train_cfg=train_cfg, test_cfg=test_cfg))
