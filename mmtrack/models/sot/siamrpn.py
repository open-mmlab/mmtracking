# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch
from mmengine import MessageHub
from torch import Tensor
from torch.nn.modules.batchnorm import BatchNorm2d, _BatchNorm
from torch.nn.modules.conv import _ConvNd

from mmtrack.registry import MODELS
from mmtrack.utils import (ForwardResults, InstanceList, OptConfigType,
                           OptMultiConfig, OptSampleList, SampleList)
from .base import BaseSingleObjectTracker


@MODELS.register_module()
class SiamRPN(BaseSingleObjectTracker):
    """SiamRPN++: Evolution of Siamese Visual Tracking with Very Deep Networks.

    This single object tracker is the implementation of `SiamRPN++
    <https://arxiv.org/abs/1812.11703>`_.
    """

    def __init__(self,
                 backbone: dict,
                 neck: Optional[dict] = None,
                 head: Optional[dict] = None,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 frozen_modules: Optional[Union[List[str], Tuple[str],
                                                str]] = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super(SiamRPN, self).__init__(data_preprocessor, init_cfg)
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        head = head.copy()
        head.update(train_cfg=train_cfg.rpn, test_cfg=test_cfg.rpn)
        self.head = MODELS.build(head)

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg

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
            for m in self.head.modules():
                if isinstance(m, _ConvNd) or isinstance(m, _BatchNorm):
                    m.reset_parameters()

    def forward_template(self, z_img: Tensor) -> Tuple[Tensor]:
        """Extract the features of exemplar images.

        Args:
            z_img (Tensor): of shape (N, C, H, W) encoding input exemplar
                images. Typically H and W equal to 127.

        Returns:
            Tuple[Tensor, ...]: Multi level feature map of exemplar
                images.
        """
        z_feat = self.backbone(z_img)
        if self.with_neck:
            z_feat = self.neck(z_feat)

        z_feat_center = []
        for i in range(len(z_feat)):
            left = (z_feat[i].size(3) - self.test_cfg.center_size) // 2
            right = left + self.test_cfg.center_size
            z_feat_center.append(z_feat[i][:, :, left:right, left:right])
        return tuple(z_feat_center)

    def forward_search(self, x_img: Tensor) -> Tuple[Tensor, ...]:
        """Extract the features of search images.

        Args:
            x_img (Tensor): of shape (N, C, H, W) encoding input search
                images. Typically H and W equal to 255.

        Returns:
            Tuple[Tensor, ...]: Multi level feature map of search images.
        """
        x_feat = self.backbone(x_img)
        if self.with_neck:
            x_feat = self.neck(x_feat)
        return x_feat

    def get_cropped_img(self, img: Tensor, center_xy: Tensor,
                        target_size: Tensor, crop_size: Tensor,
                        avg_channel: Tensor) -> Tensor:
        """Crop image.

        Only used during testing.

        This function mainly contains two steps:
        1. Crop `img` based on center `center_xy` and size `crop_size`. If the
        cropped image is out of boundary of `img`, use `avg_channel` to pad.
        2. Resize the cropped image to `target_size`.

        Args:
            img (Tensor): of shape (1, C, H, W) encoding original input
                image.
            center_xy (Tensor): of shape (2, ) denoting the center point
                for cropping image.
            target_size (int): The output size of cropped image.
            crop_size (Tensor): The size for cropping image.
            avg_channel (Tensor): of shape (3, ) denoting the padding
                values.

        Returns:
            Tensor: of shape (1, C, target_size, target_size) encoding
                the resized cropped image.
        """
        N, C, H, W = img.shape
        context_xmin = int(center_xy[0] - crop_size / 2)
        context_xmax = int(center_xy[0] + crop_size / 2)
        context_ymin = int(center_xy[1] - crop_size / 2)
        context_ymax = int(center_xy[1] + crop_size / 2)

        left_pad = max(0, -context_xmin)
        top_pad = max(0, -context_ymin)
        right_pad = max(0, context_xmax - W)
        bottom_pad = max(0, context_ymax - H)

        context_xmin += left_pad
        context_xmax += left_pad
        context_ymin += top_pad
        context_ymax += top_pad

        avg_channel = avg_channel[:, None, None]
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            new_img = img.new_zeros(N, C, H + top_pad + bottom_pad,
                                    W + left_pad + right_pad)
            new_img[..., top_pad:top_pad + H, left_pad:left_pad + W] = img
            if top_pad:
                new_img[..., :top_pad, left_pad:left_pad + W] = avg_channel
            if bottom_pad:
                new_img[..., H + top_pad:, left_pad:left_pad + W] = avg_channel
            if left_pad:
                new_img[..., :left_pad] = avg_channel
            if right_pad:
                new_img[..., W + left_pad:] = avg_channel
            crop_img = new_img[..., context_ymin:context_ymax + 1,
                               context_xmin:context_xmax + 1]
        else:
            crop_img = img[..., context_ymin:context_ymax + 1,
                           context_xmin:context_xmax + 1]

        crop_img = torch.nn.functional.interpolate(
            crop_img,
            size=(target_size, target_size),
            mode='bilinear',
            align_corners=False)
        return crop_img

    def init(self, img: Tensor) -> Tuple[Tuple[Tensor, ...], Tensor]:
        """Initialize the single object tracker in the first frame.

        Args:
            img (Tensor): of shape (1, C, H, W) encoding original input
                image.
        """
        bbox = self.memo.bbox
        z_width = bbox[2] + self.test_cfg.context_amount * (bbox[2] + bbox[3])
        z_height = bbox[3] + self.test_cfg.context_amount * (bbox[2] + bbox[3])
        z_size = torch.round(torch.sqrt(z_width * z_height))
        # used for padding when cropping the image.
        avg_channel = torch.mean(img, dim=(0, 2, 3))
        z_crop = self.get_cropped_img(img, bbox[0:2],
                                      self.test_cfg.exemplar_size, z_size,
                                      avg_channel)
        self.memo.z_feat = self.forward_template(z_crop)
        self.memo.avg_channel = avg_channel

    def track(self, img: Tensor,
              batch_data_samples: SampleList) -> InstanceList:
        """Track the box of previous frame to current frame `img`.

        Args:
            img (Tensor): of shape (1, C, H, W) encoding original input
                image.
            batch_data_samples (list[:obj:`TrackDataSample`]): The batch
                data samples. It usually includes information such
                as ``gt_instances`` and 'metainfo'.

        Returns:
            InstanceList: Tracking results of each image after the postprocess.
                - scores: a Tensor denoting the score of best_bbox.
                - bboxes: a Tensor of shape (4, ) in [x1, x2, y1, y2]
                format, and denotes the best tracked bbox in current frame.
        """
        prev_bbox = self.memo.bbox
        z_width = prev_bbox[2] + self.test_cfg.context_amount * (
            prev_bbox[2] + prev_bbox[3])
        z_height = prev_bbox[3] + self.test_cfg.context_amount * (
            prev_bbox[2] + prev_bbox[3])
        z_size = torch.sqrt(z_width * z_height)

        x_size = torch.round(
            z_size * (self.test_cfg.search_size / self.test_cfg.exemplar_size))
        x_crop = self.get_cropped_img(img, prev_bbox[0:2],
                                      self.test_cfg.search_size, x_size,
                                      self.memo.avg_channel)

        x_feat = self.forward_search(x_crop)
        scale_factor = self.test_cfg.exemplar_size / z_size

        results = self.head.predict(self.memo.z_feat, x_feat,
                                    batch_data_samples, prev_bbox,
                                    scale_factor)

        return results

    def loss(self, batch_inputs: dict, batch_data_samples: SampleList,
             **kwargs) -> dict:
        """
        Args:
            batch_inputs (Dict[str, Tensor]): of shape (N, T, C, H, W) encoding
                input images. Typically these should be mean centered and std
                scaled. The N denotes batch size. The T denotes the number of
                key/reference frames.
                - img (Tensor) : The key images.
                - ref_img (Tensor): The reference images.

            batch_data_samples (list[:obj:`TrackDataSample`]): The batch
                data samples. It usually includes information such
                as ``gt_instance``.

        Return:
            dict: A dictionary of loss components.
        """
        search_img = batch_inputs['search_img']
        assert search_img.dim(
        ) == 5, 'The img must be 5D Tensor (N, T, C, H, W).'
        search_img = search_img[:, 0]

        template_img = batch_inputs['img']
        assert template_img.dim(
        ) == 5, 'The img must be 5D Tensor (N, T, C, H, W).'
        template_img = template_img[:, 0]

        z_feat = self.forward_template(template_img)
        x_feat = self.forward_search(search_img)
        losses = self.head.loss(z_feat, x_feat, batch_data_samples, **kwargs)
        return losses

    def forward(self,
                batch_inputs: dict,
                batch_data_samples: OptSampleList = None,
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

        Note:
        the difference between this function and the ``forward`` in the
        parent class: We don't train the backbone util the
        `self.backbone_start_train_epoch`-th epoch. The epoch in this class
        counts from 0.

        Args:
            batch_inputs (Dict[str, Tensor]): The input tensor with shape
                (N, C, ...) in general.
            batch_data_samples (list[:obj:`TrackDataSample`], optional): The
                annotation data of every samples. Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`TrackDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == 'loss':
            message_hub = MessageHub.get_current_instance()
            if 'epoch' in message_hub.runtime_info:
                cur_epoch = message_hub.get_info('epoch')
                if cur_epoch >= self.train_cfg['backbone_start_train_epoch']:
                    for layer in self.train_cfg['backbone_train_layers']:
                        for param in getattr(self.backbone,
                                             layer).parameters():
                            param.requires_grad = True
                        for m in getattr(self.backbone, layer).modules():
                            if isinstance(m, BatchNorm2d):
                                m.train()
            return self.loss(batch_inputs, batch_data_samples, **kwargs)
        elif mode == 'predict':
            return self.predict(batch_inputs, batch_data_samples, **kwargs)
        elif mode == 'tensor':
            return self._forward(batch_inputs, batch_data_samples, **kwargs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')
