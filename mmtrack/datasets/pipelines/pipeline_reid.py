import cv2
import mmcv
import numpy as np
from mmcls.datasets.builder import PIPELINES
from mmcls.datasets.pipelines import ImageToTensor, ToTensor, Collect

from mmtrack.core import crop_image

@PIPELINES.register_module()
class SeqImageToTensor(ImageToTensor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, results):
        outs = []
        for _results in results:
            _results = super().__call__(_results)
            outs.append(_results)
        return outs

@PIPELINES.register_module()
class SeqToTensor(ToTensor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, results):
        outs = []
        for _results in results:
            _results = super().__call__(_results)
            outs.append(_results)
        return outs