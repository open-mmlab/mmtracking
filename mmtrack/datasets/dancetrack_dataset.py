# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.datasets import DATASETS

from .mot_challenge_dataset import MOTChallengeDataset


@DATASETS.register_module()
class DanceTrackDataset(MOTChallengeDataset):
    """Dataset for DanceTrack: https://github.com/DanceTrack/DanceTrack.

    Most content is inherited from MOTChallengeDataset.
    """

    def get_benchmark_and_eval_split(self):
        """Get benchmark and dataset split to evaluate.

        Get benchmark from upeper/lower-case image prefix and the dataset
        split to evaluate.

        Returns:
            tuple(string): The first string denotes the type of dataset.
            The second string denots the split of the dataset to eval.
        """
        SPLIT_TO_EVAL = ['train', 'val']
        for split_to_eval in SPLIT_TO_EVAL:
            if split_to_eval in self.img_prefix.upper():
                break

        return 'DanceTrack', split_to_eval
