# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

from mmtrack.registry import MODELS, TASK_UTILS
from mmtrack.utils import OptConfigType
from .deep_sort import DeepSORT


@MODELS.register_module()
class StrongSORT(DeepSORT):
    """StrongSORT: Make DeepSORT Great Again.

    Details can be found at `StrongSORT<https://arxiv.org/abs/2202.13514>`_.

    Args:
        detector (dict): Configuration of detector. Defaults to None.
        reid (dict): Configuration of reid. Defaults to None
        tracker (dict): Configuration of tracker. Defaults to None.
        kalman (dict): Configuration of Kalman filter. Defaults to None.
        cmc (dict): Configuration of camera model compensation.
            Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
           config of :class:`TrackDataPreprocessor`.  it usually includes,
            ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or list[dict]): Configuration of initialization.
            Defaults to None.
    """

    def __init__(self,
                 detector: Optional[dict] = None,
                 reid: Optional[dict] = None,
                 tracker: Optional[dict] = None,
                 kalman: Optional[dict] = None,
                 cmc: Optional[dict] = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptConfigType = None):
        super().__init__(detector, reid, tracker, kalman, data_preprocessor,
                         init_cfg)

        if kalman is not None:
            self.kalman = TASK_UTILS.build(kalman)

        if cmc is not None:
            self.cmc = TASK_UTILS.build(cmc)

    @property
    def with_cmc(self):
        """bool: whether the framework has a camera model compensation
                model.
        """
        return hasattr(self, 'cmc') and self.cmc is not None

    @property
    def with_kalman_filter(self):
        """bool: whether the framework has a Kalman filter."""
        return hasattr(self, 'kalman') and self.kalman is not None
