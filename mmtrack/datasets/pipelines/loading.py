from mmdet.datasets.pipelines import (LoadAnnotations, LoadImageFromFile,
                                      LoadMultiChannelImageFromFiles,
                                      LoadProposals)

from ..builder import PIPELINES

PIPELINES.register_module(LoadImageFromFile)
PIPELINES.register_module(LoadMultiChannelImageFromFiles)
PIPELINES.register_module(LoadAnnotations)
PIPELINES.register_module(LoadProposals)
