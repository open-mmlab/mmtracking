# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Union

import torch
from mmdet.structures.mask import mask2bbox
from mmengine.data import InstanceData
from torch import Tensor
from torch.nn import functional as F

from mmtrack.models.mot import BaseMultiObjectTracker
from mmtrack.registry import MODELS
from mmtrack.structures import TrackDataSample
from mmtrack.utils import OptConfigType, OptMultiConfig, SampleList


@MODELS.register_module()
class Mask2Former(BaseMultiObjectTracker):
    r"""Implementation of `Masked-attention Mask
    Transformer for Universal Image Segmentation
    <https://arxiv.org/pdf/2112.01527>`_."""

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

    def loss(self, batch_inputs: Dict[str, Tensor],
             batch_data_samples: SampleList, **kwargs) -> Union[dict, tuple]:
        """
        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`TrackDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        img = batch_inputs['img']
        assert img.dim() == 5, 'The img must be 5D Tensor (N, T, C, H, W).'
        img = img.flatten(0, 1)

        x = self.backbone(img)
        losses = self.track_head.loss(x, batch_data_samples)

        return losses

    def predict(self,
                batch_inputs: dict,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Dict[str, Tensor]): of shape (N, T, C, H, W)
                encoding input images. Typically these should be mean centered
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
        img = img.squeeze(1)
        feats = self.backbone(img)
        mask_cls_results, mask_pred_results = self.track_head.predict(
            feats, batch_data_samples)

        batch_img_metas = [
            data_sample.metainfo for data_sample in batch_data_samples
        ]

        results = []
        if len(mask_cls_results) > 0:
            scores = F.softmax(mask_cls_results, dim=-1)[:, :-1]
            labels = torch.arange(self.num_classes).unsqueeze(0).repeat(
                self.track_head.num_queries, 1).flatten(0, 1)
            # keep top-10 predictions
            scores_per_image, topk_indices = scores.flatten(0, 1).topk(
                10, sorted=False)
            labels_per_image = labels[topk_indices]
            topk_indices = topk_indices // self.num_classes
            mask_pred_results = mask_pred_results[topk_indices]

            img_shape = batch_img_metas[0]['img_shape']
            mask_pred_results = mask_pred_results[:, :, :img_shape[0], :
                                                  img_shape[1]]
            if rescale:
                # return result in original resolution
                ori_height, ori_width = batch_img_metas[0]['ori_shape'][:2]
                mask_pred_results = F.interpolate(
                    mask_pred_results,
                    size=(ori_height, ori_width),
                    mode='bilinear',
                    align_corners=False)

            masks = mask_pred_results > 0.

            # format top-10 predictions
            for img_idx in range(len(batch_img_metas)):
                track_data_sample = TrackDataSample()
                pred_track_instances = InstanceData()

                pred_track_instances.masks = masks[:, img_idx]
                pred_track_instances.bboxes = mask2bbox(masks[:, img_idx])
                pred_track_instances.labels = labels_per_image
                pred_track_instances.scores = scores_per_image
                pred_track_instances.instances_id = torch.arange(10)

                track_data_sample.pred_track_instances = pred_track_instances
                results.append(track_data_sample)

        return results
