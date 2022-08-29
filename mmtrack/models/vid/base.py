# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Tuple, Union

from mmengine.model import BaseModel
from torch import Tensor

from mmtrack.utils import (ForwardResults, OptConfigType, OptMultiConfig,
                           OptSampleList, SampleList)


class BaseVideoDetector(BaseModel, metaclass=ABCMeta):
    """Base class for video object detector.

    Args:
        data_preprocessor (dict or ConfigDict, optional): The pre-process
           config of :class:`TrackDataPreprocessor`.  it usually includes,
            ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or list[dict]): Initialization config dict.
    """

    def __init__(self,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(data_preprocessor, init_cfg)

    def freeze_module(self, module: Union[List[str], Tuple[str], str]) -> None:
        """Freeze module during training."""
        if isinstance(module, str):
            modules = [module]
        else:
            if not (isinstance(module, list) or isinstance(module, tuple)):
                raise TypeError('module must be a str or a list.')
            else:
                modules = module
        for module in modules:
            m = getattr(self, module)
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    @property
    def with_detector(self) -> bool:
        """bool: whether the framework has a detector"""
        return hasattr(self, 'detector') and self.detector is not None

    @property
    def with_motion(self) -> bool:
        """bool: whether the framework has a motion model"""
        return hasattr(self, 'motion') and self.motion is not None

    @property
    def with_aggregator(self) -> bool:
        """bool: whether the framework has a aggregator"""
        return hasattr(self, 'aggregator') and self.aggregator is not None

    def forward(self,
                inputs: Dict[str, Tensor],
                data_samples: OptSampleList = None,
                mode: str = 'predict',
                **kwargs) -> ForwardResults:
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`TrackDataSample`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        Args:
            inputs (Dict[str, Tensor]): of shape (N, T, C, H, W)
                encoding input images. Typically these should be mean centered
                and std scaled. The N denotes batch size. The T denotes the
                number of key/reference frames.
                - img (Tensor) : The key images.
                - ref_img (Tensor): The reference images.
            data_samples (list[:obj:`TrackDataSample`], optional): The
                annotation data of every samples. Defaults to None.
            mode (str): Return what kind of value. Defaults to 'predict'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`TrackDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == 'loss':
            return self.loss(inputs, data_samples, **kwargs)
        elif mode == 'predict':
            return self.predict(inputs, data_samples, **kwargs)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples, **kwargs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    @abstractmethod
    def loss(self, inputs: Dict[str, Tensor], data_samples: SampleList,
             **kwargs) -> Union[dict, tuple]:
        """Calculate losses from a batch of inputs and data samples."""
        pass

    @abstractmethod
    def predict(self, inputs: Dict[str, Tensor], data_samples: SampleList,
                **kwargs) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing."""
        pass

    def _forward(self,
                 inputs: Dict[str, Tensor],
                 data_samples: OptSampleList = None,
                 **kwargs):
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            inputs (Dict[str, Tensor]): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`TrackDataSample`], optional): The
                Data Samples. It usually includes information such as
                `gt_instance`.

        Returns:
            tuple[list]: A tuple of features from ``head`` forward.
        """
        raise NotImplementedError(
            "_forward function (namely 'tensor' mode) is not supported now")
