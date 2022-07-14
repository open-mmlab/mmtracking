# Copyright (c) OpenMMLab. All rights reserved.
from mmtrack.registry import DATASETS
from .mot_challenge_dataset import MOTChallengeDataset


@DATASETS.register_module()
class DanceTrackDataset(MOTChallengeDataset):
    """Dataset for DanceTrack: https://github.com/DanceTrack/DanceTrack.

    All content is inherited from MOTChallengeDataset.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
