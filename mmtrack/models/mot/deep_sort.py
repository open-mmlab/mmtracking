# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional

from torch import Tensor

from mmtrack.registry import MODELS, TASK_UTILS
from mmtrack.utils import OptConfigType, SampleList
from .base import BaseMultiObjectTracker


@MODELS.register_module()
class DeepSORT(BaseMultiObjectTracker):
    """Simple online and realtime tracking with a deep association metric.

    Details can be found at `DeepSORT<https://arxiv.org/abs/1703.07402>`_.

    Args:
        detector (dict): Configuration of detector. Defaults to None.
        reid (dict): Configuration of reid. Defaults to None
        tracker (dict): Configuration of tracker. Defaults to None.
        motion (dict): Configuration of motion. Defaults to None.
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
                 motion: Optional[dict] = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptConfigType = None):
        super().__init__(data_preprocessor, init_cfg)

        if detector is not None:
            self.detector = MODELS.build(detector)

        if reid is not None:
            self.reid = MODELS.build(reid)

        if motion is not None:
            self.motion = TASK_UTILS.build(motion)

        if tracker is not None:
            self.tracker = MODELS.build(tracker)

        self.preprocess_cfg = data_preprocessor

    def loss(self, inputs: Dict[str, Tensor], data_samples: SampleList,
             **kwargs) -> dict:
        """Calculate losses from a batch of inputs and data samples."""
        raise NotImplementedError(
            'Please train `detector` and `reid` models firstly, then \
                inference with SORT/DeepSORT.')

    def predict(self,
                inputs: Dict[str, Tensor],
                data_samples: SampleList,
                rescale: bool = True,
                **kwargs) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (Dict[str, Tensor]): of shape (N, T, C, H, W) encoding
                input images. Typically these should be mean centered and std
                scaled. The N denotes batch size.The T denotes the number of
                key/reference frames.
                - img (Tensor) : The key images.
                - ref_img (Tensor): The reference images.
            data_samples (list[:obj:`TrackDataSample`]): The batch
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
        img = inputs['img']
        assert img.dim() == 5, 'The img must be 5D Tensor (N, T, C, H, W).'
        assert img.size(0) == 1, \
            'SORT/DeepSORT inference only support ' \
            '1 batch size per gpu for now.'
        img = img[0]

        assert len(data_samples) == 1, \
            'SORT/DeepSORT inference only support ' \
            '1 batch size per gpu for now.'

        track_data_sample = data_samples[0]

        det_results = self.detector.predict(img, data_samples)
        assert len(det_results) == 1, 'Batch inference is not supported.'
        track_data_sample.pred_det_instances = det_results[
            0].pred_instances.clone()

        pred_track_instances = self.tracker.track(
            model=self,
            img=img,
            feats=None,
            data_sample=track_data_sample,
            data_preprocessor=self.preprocess_cfg,
            rescale=rescale,
            **kwargs)
        track_data_sample.pred_track_instances = pred_track_instances

        return [track_data_sample]
