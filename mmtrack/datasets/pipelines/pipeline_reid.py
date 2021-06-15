import cv2
import mmcv
import numpy as np
from mmcls.datasets.builder import PIPELINES
from mmcls.datasets.pipelines import ImageToTensor, ToTensor, Collect
from .formatting import SeqDefaultFormatBundle
from mmdet.datasets.pipelines import to_tensor
from mmcv.parallel import DataContainer as DC

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

@PIPELINES.register_module()
class SeqCollect(Collect):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, results):
        outs = []
        for _results in results:
            _results = super().__call__(_results)
            outs.append(_results)
        return outs

@PIPELINES.register_module()
class SeqReIDFormatBundle(SeqDefaultFormatBundle):
    def __init__(self, *args, **kwargs):
        super().__init__(None)

    def __call__(self, results):
        outs = []
        for _results in results:
            _results = self.default_format_bundle(_results)
            _results = self.reid_format_bundle(_results)
            outs.append(_results)

        return outs

    def reid_format_bundle(self, results):
        key = 'gt_label'
        results[key] = DC(to_tensor(results[key]), stack=True, pad_dims=None)
        return results
