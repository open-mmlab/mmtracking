# Copyright (c) OpenMMLab. All rights reserved.
from .inference import inference_mot, inference_sot, inference_vid, init_model

__all__ = ['init_model', 'inference_mot', 'inference_sot', 'inference_vid']
