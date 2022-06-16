# Copyright (c) OpenMMLab. All rights reserved.
from mmcls.models.classifiers import ImageClassifier

from mmtrack.registry import MODELS


@MODELS.register_module()
class BaseReID(ImageClassifier):
    """Base model for re-identification."""

    pass
