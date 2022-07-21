# Copyright (c) OpenMMLab. All rights reserved.
import math
from copy import deepcopy
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from mmdet.structures.bbox.transforms import bbox_xyxy_to_cxcywh
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvNd

from mmtrack.registry import MODELS
from mmtrack.structures import TrackDataSample
from mmtrack.utils import (InstanceList, OptConfigType, OptMultiConfig,
                           SampleList)
from .base import BaseSingleObjectTracker


@MODELS.register_module()
class Stark(BaseSingleObjectTracker):
    """STARK: Learning Spatio-Temporal Transformer for Visual Tracking.

    This single object tracker is the implementation of `STARk
    <https://arxiv.org/abs/2103.17154>`_.

    Args:
        backbone (dict): the configuration of backbone network.
        neck (dict, optional): the configuration of neck network.
            Defaults to None.
        head (dict, optional): the configuration of head network.
            Defaults to None.
        init_cfg (dict, optional): the configuration of initialization.
            Defaults to None.
        frozen_modules (str | list | tuple, optional): the names of frozen
            modules. Defaults to None.
        train_cfg (dict, optional): the configuratioin of train.
            Defaults to None.
        test_cfg (dict, optional): the configuration of test.
            Defaults to None.
    """

    def __init__(self,
                 backbone: dict,
                 neck: Optional[dict] = None,
                 head: Optional[dict] = None,
                 pretrains: Optional[dict] = None,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 frozen_modules: Optional[Union[List[str], Tuple[str],
                                                str]] = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super(Stark, self).__init__(data_preprocessor, init_cfg)
        head.update(test_cfg=test_cfg)
        self.backbone = MODELS.build(backbone)
        self.neck = MODELS.build(neck)
        self.head = MODELS.build(head)

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg

        self.num_templates = self.test_cfg['num_templates']

        # Set the update interval
        self.update_intervals = self.test_cfg.get('update_intervals', None)
        if isinstance(self.update_intervals, (int, float)):
            self.update_intervals = [int(self.update_intervals)
                                     ] * self.num_templates

        if frozen_modules is not None:
            self.freeze_module(frozen_modules)

    def init_weights(self):
        """Initialize the weights of modules in single object tracker."""
        # We don't use the `init_weights()` function in BaseModule, since it
        # doesn't support the initialization method from `reset_parameters()`
        # in Pytorch.
        if self.with_backbone:
            self.backbone.init_weights()

        if self.with_neck:
            for m in self.neck.modules():
                if isinstance(m, _ConvNd) or isinstance(m, _BatchNorm):
                    m.reset_parameters()

        if self.with_head:
            self.head.init_weights()

    def extract_feat(self, img: Tensor) -> Tensor:
        """Extract the features of the input image.

        Args:
            img (Tensor): image of shape (N, C, H, W).

        Returns:
            tuple(Tensor): the multi-level feature maps, and each of them is
                    of shape (N, C, H // stride, W // stride).
        """
        feat = self.backbone(img)
        feat = self.neck(feat)
        return feat

    def get_cropped_img(self, img: Tensor, target_bbox: Tensor,
                        search_area_factor: float,
                        output_size: float) -> Union[Tensor, float, Tensor]:
        """ Crop Image
        Only used during testing
        This function mainly contains two steps:
        1. Crop `img` based on target_bbox and search_area_factor. If the
        cropped image/mask is out of boundary of `img`, use 0 to pad.
        2. Resize the cropped image/mask to `output_size`.

        args:
            img (Tensor): of shape (1, C, H, W)
            target_bbox (Tensor): in [cx, cy, w, h] format
            search_area_factor (float): Ratio of crop size to target size.
            output_size (float): the size of output cropped image
                (always square).
        returns:
            img_crop_padded (Tensor): of shape (1, C, output_size, output_size)
            resize_factor (float): the ratio of original image scale to cropped
                image scale.
            pdding_mask (Tensor): the padding mask caused by cropping. It's
                of shape (1, output_size, output_size).
        """
        cx, cy, w, h = target_bbox.split((1, 1, 1, 1), dim=-1)

        img_h, img_w = img.shape[2:]
        # 1. Crop image
        # 1.1 calculate crop size and pad size
        crop_size = math.ceil(math.sqrt(w * h) * search_area_factor)
        if crop_size < 1:
            raise Exception('Too small bounding box.')

        x1 = torch.round(cx - crop_size * 0.5).long()
        x2 = x1 + crop_size
        y1 = torch.round(cy - crop_size * 0.5).long()
        y2 = y1 + crop_size

        x1_pad = max(0, -x1)
        x2_pad = max(x2 - img_w + 1, 0)
        y1_pad = max(0, -y1)
        y2_pad = max(y2 - img_h + 1, 0)

        # 1.2 crop image
        img_crop = img[..., y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad]

        # 1.3 pad image
        img_crop_padded = F.pad(
            img_crop,
            pad=(x1_pad, x2_pad, y1_pad, y2_pad),
            mode='constant',
            value=0)
        # 1.4 generate padding mask
        _, _, img_h, img_w = img_crop_padded.shape
        end_x = None if x2_pad == 0 else -x2_pad
        end_y = None if y2_pad == 0 else -y2_pad
        padding_mask = torch.ones((img_h, img_w),
                                  dtype=torch.float32,
                                  device=img.device)
        padding_mask[y1_pad:end_y, x1_pad:end_x] = 0.

        # 2. Resize cropped image and padding mask
        resize_factor = output_size / crop_size
        img_crop_padded = F.interpolate(
            img_crop_padded, (output_size, output_size),
            mode='bilinear',
            align_corners=False)

        padding_mask = F.interpolate(
            padding_mask[None, None], (output_size, output_size),
            mode='bilinear',
            align_corners=False).squeeze(dim=0).type(torch.bool)

        return img_crop_padded, resize_factor, padding_mask

    def init(self, img: Tensor):
        """Initialize the single object tracker in the first frame.

        Args:
            img (Tensor): input image of shape (1, C, H, W).
        """
        self.memo.z_dict_list = []  # store templates
        # get the 1st template
        z_patch, _, z_mask = self.get_cropped_img(
            img, self.memo.bbox, self.test_cfg['template_factor'],
            self.test_cfg['template_size']
        )  # z_patch of shape [1,C,H,W];  z_mask of shape [1,H,W]

        z_feat = self.extract_feat(z_patch)

        self.z_dict = dict(feat=z_feat, mask=z_mask)
        self.memo.z_dict_list.append(self.z_dict)

        # get other templates
        for _ in range(self.num_templates - 1):
            self.memo.z_dict_list.append(deepcopy(self.z_dict))

    def update_template(self, img: Tensor, bbox: Union[List, Tensor],
                        conf_score: float):
        """Update the dymanic templates.

        Args:
            img (Tensor): of shape (1, C, H, W).
            bbox (list | Tensor): in [cx, cy, w, h] format.
            conf_score (float): the confidence score of the predicted bbox.
        """
        for i, update_interval in enumerate(self.update_intervals):
            if self.memo.frame_id % update_interval == 0 and conf_score > 0.5:
                z_patch, _, z_mask = self.get_cropped_img(
                    img,
                    bbox,
                    self.test_cfg['template_factor'],
                    output_size=self.test_cfg['template_size'])
                z_feat = self.extract_feat(z_patch)
                # the 1st element of z_dict_list is the template from the 1st
                # frame
                self.memo.z_dict_list[i + 1] = dict(feat=z_feat, mask=z_mask)

    def track(self, img: Tensor,
              batch_data_samples: SampleList) -> InstanceList:
        """Track the box of previous frame to current frame `img`.

        Args:
            img (Tensor): of shape (1, C, H, W).
            batch_data_samples (list[:obj:`TrackDataSample`]): The batch
                data samples. It usually includes information such
                as ``gt_instances`` and 'metainfo'.

        Returns:
            InstanceList: Tracking results of each image after the postprocess.
                - scores: a Tensor denoting the score of best_bbox.
                - bboxes: a Tensor of shape (4, ) in [x1, x2, y1, y2]
                format, and denotes the best tracked bbox in current frame.
        """
        # get the search patches
        x_patch, resize_factor, x_mask = self.get_cropped_img(
            img, self.memo.bbox, self.test_cfg['search_factor'],
            self.test_cfg['search_size']
        )  # bbox: of shape (x1, y1, w, h), x_mask: of shape (1, h, w)

        x_feat = self.extract_feat(x_patch)
        x_dict = dict(feat=x_feat, mask=x_mask)
        head_inputs = self.memo.z_dict_list + [x_dict]
        results = self.head.predict(head_inputs, batch_data_samples,
                                    self.memo.bbox, resize_factor)

        if results[0].scores.item() != -1:
            # get confidence score (whether the search region is reliable)
            crop_bbox = bbox_xyxy_to_cxcywh(results[0].bboxes.squeeze())
            self.update_template(img, crop_bbox, results[0].scores.item())

        return results

    def predict_vot(self, batch_inputs: dict,
                    batch_data_samples: List[TrackDataSample]):
        raise NotImplementedError(
            'STARK does not support testing on VOT datasets yet.')

    def loss(self, batch_inputs: dict,
             batch_data_samples: List[TrackDataSample], **kwargs) -> dict:
        """Forward of training.

        Args:
            batch_inputs (dict[Tensor]): of shape (N, T, C, H, W) encoding
                input images. Typically these should be mean centered and std
                scaled. The N denotes batch size. The T denotes the number of
                key/reference frames.
                - img (Tensor) : The key images.
                - ref_img (Tensor): The reference images.

            batch_data_samples (list[:obj:`TrackDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance`.

        Return:
            dict: A dictionary of loss components.
        """
        template_padding_mask = [
            data_sample.padding_mask for data_sample in batch_data_samples
        ]
        template_padding_mask = torch.stack(template_padding_mask, dim=0)
        search_padding_mask = [
            data_sample.search_padding_mask
            for data_sample in batch_data_samples
        ]
        search_padding_mask = torch.stack(search_padding_mask, dim=0)

        search_img = batch_inputs['search_img']
        assert search_img.dim(
        ) == 5, 'The img must be 5D Tensor (N, T, C, H, W).'
        template_img = batch_inputs['img']
        assert template_img.dim(
        ) == 5, 'The img must be 5D Tensor (N, T, C, H, W).'

        head_inputs = []
        for i in range(self.num_templates):
            z_feat = self.extract_feat(template_img[:, i])
            z_dict = dict(feat=z_feat, mask=template_padding_mask[:, i])
            head_inputs.append(z_dict)
        x_feat = self.extract_feat(search_img[:, 0])
        x_dict = dict(feat=x_feat, mask=search_padding_mask[:, 0])
        head_inputs.append(x_dict)

        losses = self.head.loss(head_inputs, batch_data_samples)

        return losses
