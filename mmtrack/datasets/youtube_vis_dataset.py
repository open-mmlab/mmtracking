# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.datasets import DATASETS

from .coco_video_dataset import CocoVideoDataset


@DATASETS.register_module()
class YouTubeVISDataset(CocoVideoDataset):
    """YouTube VIS dataset for video instance segmentation."""

    CLASSES_2019_version = ('person', 'giant_panda', 'lizard', 'parrot',
                            'skateboard', 'sedan', 'ape', 'dog', 'snake',
                            'monkey', 'hand', 'rabbit', 'duck', 'cat', 'cow',
                            'fish', 'train', 'horse', 'turtle', 'bear',
                            'motorbike', 'giraffe', 'leopard', 'fox', 'deer',
                            'owl', 'surfboard', 'airplane', 'truck', 'zebra',
                            'tiger', 'elephant', 'snowboard', 'boat', 'shark',
                            'mouse', 'frog', 'eagle', 'earless_seal',
                            'tennis_racket')

    CLASSES_2021_version = ('airplane', 'bear', 'bird', 'boat', 'car', 'cat',
                            'cow', 'deer', 'dog', 'duck', 'earless_seal',
                            'elephant', 'fish', 'flying_disc', 'fox', 'frog',
                            'giant_panda', 'giraffe', 'horse', 'leopard',
                            'lizard', 'monkey', 'motorbike', 'mouse', 'parrot',
                            'person', 'rabbit', 'shark', 'skateboard', 'snake',
                            'snowboard', 'squirrel', 'surfboard',
                            'tennis_racket', 'tiger', 'train', 'truck',
                            'turtle', 'whale', 'zebra')

    def __init__(self, dataset_version, *args, **kwargs):
        self.set_dataset_classes(dataset_version)
        super().__init__(*args, **kwargs)

    @classmethod
    def set_dataset_classes(cls, dataset_version):
        if dataset_version == '2019':
            cls.CLASSES = cls.CLASSES_2019_version
        elif dataset_version == '2021':
            cls.CLASSES = cls.CLASSES_2021_version
        else:
            raise NotImplementedError('Not supported YouTubeVIS dataset'
                                      f'version: {dataset_version}')
