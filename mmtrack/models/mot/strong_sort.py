# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional

from torch import Tensor

from mmtrack.utils import OptConfigType, SampleList
from mmtrack.registry import MODELS, TASK_UTILS
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
        super().__init__(data_preprocessor=data_preprocessor,
                         init_cfg=init_cfg)

        if detector is not None:
            self.detector = MODELS.build(detector)

        if reid is not None:
            self.reid = MODELS.build(reid)

        if kalman is not None:
            self.kalman = TASK_UTILS.build(kalman)

        if cmc is not None:
            self.cmc = TASK_UTILS.build(cmc)

        if tracker is not None:
            self.tracker = MODELS.build(tracker)

        self.preprocess_cfg = data_preprocessor

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

    def predict(self,
                batch_inputs: Dict[str, Tensor],
                batch_data_samples: SampleList,
                rescale: bool = True,
                **kwargs) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Dict[str, Tensor]): of shape (N, T, C, H, W) encoding
                input images. Typically these should be mean centered and std
                scaled. The N denotes batch size.The T denotes the number of
                key/reference frames.
                - img (Tensor) : The key images.
                - ref_img (Tensor): The reference images.
            batch_data_samples (list[:obj:`TrackDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance`.
            rescale (bool, Optional): If False, then returned bboxes and masks
                will fit the scale of img, otherwise, returned bboxes and masks
                will fit the scale of original image shape. Defaults to True.

        Returns:
            SampleList: Tracking results of the
                input images. Each TrackDataSample usually contains
                ``pred_det_instances`` or ``pred_track_instances``.
        """
        img = batch_inputs['img']
        assert img.dim() == 5, 'The img must be 5D Tensor (N, T, C, H, W).'
        assert img.size(0) == 1, \
            'StrongSORT inference only support ' \
            '1 batch size per gpu for now.'

        assert len(batch_data_samples) == 1, \
            'StrongSORT inference only support ' \
            '1 batch size per gpu for now.'

        return super().predict(
            batch_inputs, batch_data_samples, rescale, **kwargs)
