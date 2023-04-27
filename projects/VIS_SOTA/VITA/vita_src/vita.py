# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Dict, Optional, Union

import torch
from torch import Tensor

from mmtrack.models.mot import BaseMultiObjectTracker
from mmtrack.registry import MODELS
from mmtrack.utils import OptConfigType, OptMultiConfig, SampleList


@MODELS.register_module()
class VITA(BaseMultiObjectTracker):

    def __init__(self,
                 backbone: Optional[dict] = None,
                 seg_head: Optional[dict] = None,
                 track_head: Optional[dict] = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(data_preprocessor, init_cfg)

        seg_head_ = seg_head.deepcopy()
        track_head_ = track_head.deepcopy()
        seg_head_.update(train_cfg=train_cfg)
        track_head_.update(test_cfg=test_cfg)

        if backbone is not None:
            self.backbone = MODELS.build(backbone)

        if seg_head is not None:
            self.seg_head = MODELS.build(seg_head_)

        if track_head is not None:
            self.track_head = MODELS.build(track_head_)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def loss(self, inputs: Dict[str, Tensor], data_samples: SampleList,
             **kwargs) -> Union[dict, tuple]:
        """
        Args:
            inputs (Tensor): Input images of shape (N, T, C, H, W).
                These should usually be mean centered and std scaled.
            data_samples (list[:obj:`TrackDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        img = inputs['img']
        assert img.dim() == 5, 'The img must be 5D Tensor (N, T, C, H, W).'
        # shape (N * T, C, H, W)
        img = img.flatten(0, 1)

        feats = self.backbone(img)
        losses = self.seg_head.loss(feats, data_samples)

        return losses

    def predict(self,
                inputs: dict,
                data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with
        postprocessing.

        Args:
            inputs (Dict[str, Tensor]): of shape (N, T, C, H, W)
                encoding input images. Typically, these should be mean centered
                and std scaled. The N denotes batch size. The T denotes the
                number of key/reference frames.
                - img (Tensor) : The key images.
                - ref_img (Tensor): The reference images.
                In test mode, T = 1 and there is only ``img`` and no
                ``ref_img``.
            data_samples (list[:obj:`TrackDataSample`]): The batch
                data samples. It usually includes information such
                as ``gt_instances`` and 'metainfo'.
            rescale (bool, Optional): If False, then returned bboxes and masks
                will fit the scale of img, otherwise, returned bboxes and masks
                will fit the scale of original image shape. Defaults to True.

        Returns:
            SampleList: Tracking results of the input images.
            Each TrackDataSample usually contains ``pred_track_instances``.
        """
        img = inputs['img']
        assert img.dim() == 5, 'The img must be 5D Tensor (N, T, C, H, W).'
        # the "T" is 1
        img = img.squeeze(1)
        num_frames = img.size(0)
        frame_queries, mask_features = [], []

        test_run_chunk_size = self.test_cfg.get('test_run_chunk_size', 18)
        for i in range(math.ceil(num_frames / test_run_chunk_size)):
            clip_imgs = img[i * test_run_chunk_size:(i + 1) *
                            test_run_chunk_size]

            feats = self.backbone(clip_imgs)
            _frame_queries, _mask_features = self.seg_head.predict(feats)
            # just a conv2d
            _mask_features = self.track_head.vita_mask_features(_mask_features)

            frame_queries.append(_frame_queries)
            mask_features.append(_mask_features)

        frame_queries = torch.cat(frame_queries)[None]
        mask_features = torch.cat(mask_features)
        pred_track_ins_list = self.track_head.predict(mask_features,
                                                      frame_queries,
                                                      data_samples, rescale)

        results = []
        for idx, pred_track_ins in enumerate(pred_track_ins_list):
            track_data_sample = data_samples[idx]
            track_data_sample.pred_track_instances = pred_track_ins
            results.append(track_data_sample)

        return results
