# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Tuple, Union

from torch import Tensor

from mmtrack.models.mot import BaseMultiObjectTracker
from mmtrack.registry import MODELS
from mmtrack.utils import OptConfigType, OptMultiConfig, SampleList


@MODELS.register_module()
class IDOL(BaseMultiObjectTracker):

    def __init__(self,
                 backbone: Optional[dict] = None,
                 neck: Optional[dict] = None,
                 track_head: Optional[dict] = None,
                 tracker: Optional[dict] = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(data_preprocessor, init_cfg)

        if backbone is not None:
            self.backbone = MODELS.build(backbone)

        if neck is not None:
            self.neck = MODELS.build(neck)

        if track_head is not None:
            track_head.update(train_cfg=train_cfg)
            track_head.update(test_cfg=test_cfg)
            self.track_head = MODELS.build(track_head)

        if tracker is not None:
            self.tracker = MODELS.build(tracker)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.cur_train_mode = train_cfg.cur_train_mode

    def loss(self, inputs: Dict[str, Tensor], data_samples: SampleList,
             **kwargs) -> Union[dict, tuple]:

        img = inputs['img']
        assert img.dim() == 5, 'The img must be 5D Tensor (N, T, C, H, W).'
        # shape (N * T, C, H, W)
        img = img.flatten(0, 1)

        x = self.extract_feat(img)
        losses = self.track_head.loss(x, data_samples)

        return losses

    def predict(self,
                inputs: dict,
                data_samples: SampleList,
                rescale: bool = True) -> SampleList:

        img = inputs['img']
        assert img.dim() == 5, 'The img must be 5D Tensor (N, T, C, H, W).'
        # the "T" is 1
        img = img.squeeze(1)
        feats = self.extract_feat(img)
        pred_det_ins_list = self.track_head.predict(feats, data_samples,
                                                    rescale)
        track_data_sample = data_samples[0]
        pred_det_ins = pred_det_ins_list[0]
        track_data_sample.pred_det_instances = \
            pred_det_ins.clone()

        if self.cur_train_mode == 'VIS':
            pred_track_instances = self.tracker.track(
                data_sample=track_data_sample, rescale=rescale)
            track_data_sample.pred_track_instances = pred_track_instances
        else:
            pred_det_ins.masks = pred_det_ins.masks.squeeze(1) > 0.5
            track_data_sample.pred_instances = pred_det_ins

        return [track_data_sample]

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N * T, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        x = self.backbone(batch_inputs)
        x = self.neck(x)
        return x
