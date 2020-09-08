from mmcv.utils import Registry
from mmdet.models.builder import build

MODELS = Registry('model')
TRACKERS = Registry('tracker')


def build_tracker(cfg):
    """Build tracker."""
    return build(cfg, TRACKERS)


def build_model(cfg, train_cfg=None, test_cfg=None):
    """Build model."""
    return build(cfg, MODELS, dict(train_cfg=train_cfg, test_cfg=test_cfg))


def register_from_mmdet(model):
    import mmdet
    model = getattr(mmdet.models, model)
    MODELS.register_module(module=model)
