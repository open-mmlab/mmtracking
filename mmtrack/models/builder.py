from mmcv.utils import Registry
from mmdet.models import DETECTORS
from mmdet.models.builder import build

MODELS = Registry('model')
TRACKERS = Registry('tracker')
MOTION = Registry('motion')


def build_tracker(cfg):
    """Build tracker."""
    return build(cfg, TRACKERS)


def build_motion(cfg):
    """Build motion model."""
    return build(cfg, MOTION)


def build_detector(cfg):
    """Build detector."""
    return build(cfg, DETECTORS)


def build_model(cfg):
    """Build model."""
    return build(cfg, MODELS)
