# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Union

from torch import Tensor

from mmtrack.models.mot import BaseMultiObjectTracker
from mmtrack.registry import MODELS
from mmtrack.structures import TrackDataSample
from mmtrack.utils import OptConfigType, OptMultiConfig, SampleList


@MODELS.register_module()
class Mask2Former(BaseMultiObjectTracker):
    r"""Implementation of `Masked-attention Mask
    Transformer for Universal Image Segmentation
    <https://arxiv.org/pdf/2112.01527>`_.

    Args:
        backbone (dict): Configuration of backbone. Defaults to None.
        track_head (dict): Configuration of track head. Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
           config of :class:`TrackDataPreprocessor`.  it usually includes,
            ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
            Defaults to None.
        init_cfg (dict or list[dict]): Configuration of initialization.
            Defaults to None.
    """

    def __init__(self,
                 backbone: Optional[dict] = None,
                 track_head: Optional[dict] = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super(BaseMultiObjectTracker, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        if backbone is not None:
            self.backbone = MODELS.build(backbone)

        if track_head is not None:
            self.track_head = MODELS.build(track_head)

        self.num_classes = self.track_head.num_classes

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """Overload in order to load mmdet pretrained ckpt."""
        for key in list(state_dict):
            if key.startswith('panoptic_head'):
                state_dict[key.replace('panoptic',
                                       'track')] = state_dict.pop(key)

        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    def loss(self, batch_inputs: Dict[str, Tensor],
             batch_data_samples: SampleList, **kwargs) -> Union[dict, tuple]:
        """
        Args:
            batch_inputs (Tensor): Input images of shape (N, T, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`TrackDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        img = batch_inputs['img']
        assert img.dim() == 5, 'The img must be 5D Tensor (N, T, C, H, W).'
        # shape (N * T, C, H, W)
        img = img.flatten(0, 1)

        x = self.backbone(img)
        losses = self.track_head.loss(x, batch_data_samples)

        return losses

    def predict(self,
                batch_inputs: dict,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with
        postprocessing.

        Args:
            batch_inputs (Dict[str, Tensor]): of shape (N, T, C, H, W)
                encoding input images. Typically, these should be mean centered
                and std scaled. The N denotes batch size. The T denotes the
                number of key/reference frames.
                - img (Tensor) : The key images.
                - ref_img (Tensor): The reference images.
                In test mode, T = 1 and there is only ``img`` and no
                ``ref_img``.
            batch_data_samples (list[:obj:`TrackDataSample`]): The batch
                data samples. It usually includes information such
                as ``gt_instances`` and 'metainfo'.
            rescale (bool, Optional): If False, then returned bboxes and masks
                will fit the scale of img, otherwise, returned bboxes and masks
                will fit the scale of original image shape. Defaults to True.

        Returns:
            SampleList: Tracking results of the input images.
            Each TrackDataSample usually contains ``pred_track_instances``.
        """
        img = batch_inputs['img']
        assert img.dim() == 5, 'The img must be 5D Tensor (N, T, C, H, W).'
        # the "T" is 1
        img = img.squeeze(1)
        feats = self.backbone(img)
        pred_track_ins_list = self.track_head.predict(feats,
                                                      batch_data_samples,
                                                      rescale)

        results = []
        for pred_track_ins in pred_track_ins_list:
            track_data_sample = TrackDataSample()
            track_data_sample.pred_track_instances = pred_track_ins
            results.append(track_data_sample)

        return results
