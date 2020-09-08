from .builder import MODELS, TRACKERS, build_model, build_tracker
from .losses import *  # noqa: F401,F403
from .MOT import *  # noqa: F401,F403
from .roi_heads import *  # noqa: F401,F403
from .SOT import *  # noqa: F401,F403
from .trackers import *  # noqa: F401,F403
from .VID import *  # noqa: F401,F403

from mmdet.models import FasterRCNN
MODELS.register_module(module=FasterRCNN)

__all__ = ['MODELS', 'TRACKERS', 'build_model', 'build_tracker']
