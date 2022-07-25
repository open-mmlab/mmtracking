# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Union

from mmtrack.registry import DATASETS
from .base_video_dataset import BaseVideoDataset


@DATASETS.register_module()
class YouTubeVISDataset(BaseVideoDataset):
    """YouTube VIS dataset for video instance segmentation.

    Args:
        dataset_version (str): Select dataset year version.
    """

    def __init__(self, dataset_version: str, *args, **kwargs):
        self.set_dataset_classes(dataset_version)
        super().__init__(*args, **kwargs)

    @classmethod
    def set_dataset_classes(cls, dataset_version: str) -> None:
        """Pass the category of the corresponding year to metainfo.

        Args:
            dataset_version (str): Select dataset year version.
        """
        CLASSES_2019_version = ('person', 'giant_panda', 'lizard', 'parrot',
                                'skateboard', 'sedan', 'ape', 'dog', 'snake',
                                'monkey', 'hand', 'rabbit', 'duck', 'cat',
                                'cow', 'fish', 'train', 'horse', 'turtle',
                                'bear', 'motorbike', 'giraffe', 'leopard',
                                'fox', 'deer', 'owl', 'surfboard', 'airplane',
                                'truck', 'zebra', 'tiger', 'elephant',
                                'snowboard', 'boat', 'shark', 'mouse', 'frog',
                                'eagle', 'earless_seal', 'tennis_racket')

        CLASSES_2021_version = ('airplane', 'bear', 'bird', 'boat', 'car',
                                'cat', 'cow', 'deer', 'dog', 'duck',
                                'earless_seal', 'elephant', 'fish',
                                'flying_disc', 'fox', 'frog', 'giant_panda',
                                'giraffe', 'horse', 'leopard', 'lizard',
                                'monkey', 'motorbike', 'mouse', 'parrot',
                                'person', 'rabbit', 'shark', 'skateboard',
                                'snake', 'snowboard', 'squirrel', 'surfboard',
                                'tennis_racket', 'tiger', 'train', 'truck',
                                'turtle', 'whale', 'zebra')

        if dataset_version == '2019':
            cls.METAINFO = dict(CLASSES=CLASSES_2019_version)
        elif dataset_version == '2021':
            cls.METAINFO = dict(CLASSES=CLASSES_2021_version)
        else:
            raise NotImplementedError('Not supported YouTubeVIS dataset'
                                      f'version: {dataset_version}')

    def __getitem__(self, idx: Union[int, list]) -> Union[dict, list]:
        """Get the idx-th image and data information of dataset after
        ``self.pipeline``, and ``full_init`` will be called if the dataset has
        not been fully initialized.

        During training phase, if ``self.pipeline`` get ``None``,
        ``self._rand_another`` will be called until a valid image is fetched or
         the maximum limit of refetech is reached.

        Args:
            idx (int or list): The index of self.data_list.

        Returns:
            dict: The idx-th image and data information of dataset after
            ``self.pipeline``.
        """
        # Performing full initialization by calling `__getitem__` will consume
        # extra memory. If a dataset is not fully initialized by setting
        # `lazy_init=True` and then fed into the dataloader. Different workers
        # will simultaneously read and parse the annotation. It will cost more
        # time and memory, although this may work. Therefore, it is recommended
        # to manually call `full_init` before dataset fed into dataloader to
        # ensure all workers use shared RAM from master process.
        if not self._fully_initialized:
            warnings.warn(
                'Please call `full_init()` method manually to accelerate '
                'the speed.')
            self.full_init()

        if self.test_mode:
            # support to read all frames in one video
            if isinstance(idx, list):
                data_list = []
                for _idx in idx:
                    data = self.prepare_data(_idx)
                    if data is None:
                        raise Exception(
                            'Test time pipline should not get `None` '
                            'data_sample')
                    data_list.append(data)
                return data_list
            else:
                data = self.prepare_data(idx)
                if data is None:
                    raise Exception('Test time pipline should not get `None` '
                                    'data_sample')
                return data

        for _ in range(self.max_refetch + 1):
            data = self.prepare_data(idx)
            # Broken images or random augmentations may cause the returned data
            # to be None
            if data is None:
                idx = self._rand_another()
                continue
            return data

        raise Exception(f'Cannot find valid image after {self.max_refetch}! '
                        'Please check your image path and pipeline')
