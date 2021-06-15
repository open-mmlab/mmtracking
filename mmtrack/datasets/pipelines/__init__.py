from mmdet.datasets.builder import PIPELINES

from .formatting import (ConcatVideoReferences, SeqDefaultFormatBundle, ToList,
                         VideoCollect)
from .loading import (LoadDetections, LoadMultiImagesFromFile,
                      SeqLoadAnnotations)
from .processing import MatchInstances
from .transforms import (SeqBlurAug, SeqColorAug, SeqCropLikeSiamFC,
                         SeqNormalize, SeqPad, SeqPhotoMetricDistortion,
                         SeqRandomCrop, SeqRandomFlip, SeqResize,
                         SeqShiftScaleAug)

from .pipeline_reid import (SeqImageToTensor, SeqToTensor, SeqCollect, SeqReIDFormatBundle)

__all__ = [
    'PIPELINES', 'LoadMultiImagesFromFile', 'SeqLoadAnnotations', 'SeqResize',
    'SeqNormalize', 'SeqRandomFlip', 'SeqPad', 'SeqDefaultFormatBundle',
    'VideoCollect', 'ConcatVideoReferences', 'LoadDetections',
    'MatchInstances', 'SeqRandomCrop', 'SeqPhotoMetricDistortion',
    'SeqCropLikeSiamFC', 'SeqShiftScaleAug', 'SeqBlurAug', 'SeqColorAug',
    'ToList', 'SeqImageToTensor', 'SeqToTensor', 'SeqCollect', 'SeqReIDFormatBundle'
]
