# Copyright (c) OpenMMLab. All rights reserved.
"""Collecting some commonly used type hint in mmdetection."""
from typing import Dict, List, Optional, Tuple, Union

import torch
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData

from ..structures import TrackDataSample

# Type hint of config data
ConfigType = Union[ConfigDict, dict]
OptConfigType = Optional[ConfigType]
# Type hint of one or more config data
MultiConfig = Union[ConfigType, List[ConfigType]]
OptMultiConfig = Optional[MultiConfig]

InstanceList = List[InstanceData]
OptInstanceList = Optional[InstanceList]

SampleList = List[TrackDataSample]
OptSampleList = Optional[SampleList]

ForwardResults = Union[Dict[str, torch.Tensor], List[TrackDataSample],
                       Tuple[torch.Tensor], torch.Tensor]
