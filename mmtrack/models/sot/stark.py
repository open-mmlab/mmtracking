# Copyright (c) OpenMMLab. All rights reserved.
from collections import defaultdict

import torch
import torch.nn.functional as F
from mmdet.models.builder import build_backbone, build_head, build_neck
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvNd

from ..builder import MODELS
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
                 backbone,
                 neck=None,
                 head=None,
                 init_cfg=None,
                 frozen_modules=None,
                 train_cfg=None,
                 test_cfg=None):
        super(Stark, self).__init__(init_cfg)
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        self.head = build_head(head)

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg

        # Set the update interval
        self.update_intervals = self.test_cfg['update_intervals']
        self.num_extra_template = len(self.update_intervals)

        self.debug = False
        if frozen_modules is not None:
            self.freeze_module(frozen_modules)
        # We only train classification head in stage-2. The transformer and
        # bbox head are not trained in stage-2.
        if self.head.run_cls_head and not self.head.run_bbox_head:
            for name, module in self.head.named_parameters():
                if 'cls_head' not in name:
                    module.requires_grad = False

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

    def _merge_template_search(self, inputs):
        """Merge the data of template and search images.
        The merge includes 3 steps: flatten, premute and concatenate.
        Note: the data of search image must be in the last place.

        args:
            inputs (list[dict(Tensor)]):
                The list contains the data of template and search images.
                The dict is in the following format:
                - 'feat': (1, C, H, W)
                - 'mask': (1, H, W)
                - 'pos_embed': (1, C, H, W)

        Return:
            dict(Tensor):
                - 'feat': in [data_flatten_len, 1, C] format
                - 'mask': in [1, data_flatten_len] format
                - 'pos_embed': in [data_flatten_len, 1, C]
                    format

                Here, 'data_flatten_len' = z_h*z_w*2 + x_h*x_w.
                'z_h' and 'z_w' denote the height and width of the
                template images respectively.
                'x_h' and 'x_w' denote the height and width of search image
                respectively.
        """
        seq_dict = defaultdict(list)
        # flatten and permute
        for input_dic in inputs:
            for name, x in input_dic.items():
                if name == 'mask':
                    seq_dict[name].append(x.flatten(1))
                else:
                    seq_dict[name].append(
                        x.flatten(2).permute(2, 0, 1).contiguous())
        # concatenate
        for name, x in seq_dict.items():
            if name == 'mask':
                seq_dict[name] = torch.cat(x, dim=1)
            else:
                seq_dict[name] = torch.cat(x, dim=0)
        return seq_dict

    def forward_before_head(self, img, mask):
        """Extract the features of the input image and resize mask to the shape
        of features.

        Args:
            img (Tensor): image of shape (N, C, H, W).
            mask (Tensor): mask of shape (N, H, W).

        Returns:
            dict(Tensor):
                - 'feat': the multi-level feature map of shape
                    (N, C, H // stride, W // stride).
                - 'mask': the mask of shape (N, H // stride, W // stride).
                - 'pos_embed': of the position embedding of shape
                    (N, C, H // stride, W // stride)
        """
        feat = self.backbone(img)
        feat = self.neck(feat)[0]

        # official code uses default 'bilinear' interpolation.
        mask = F.interpolate(
            mask[None].float(), size=feat.shape[-2:]).to(torch.bool)[0]
        pos_embed = self.head.positional_encoding(mask)

        return {'feat': feat, 'mask': mask, 'pos_embed': pos_embed}

    def forward_train(self,
                      img,
                      img_metas,
                      search_img,
                      search_img_metas,
                      gt_bboxes,
                      padding_mask,
                      search_gt_bboxes,
                      search_padding_mask,
                      search_gt_labels=None,
                      **kwargs):
        """forward of training.

        Args:
            img (Tensor): template images of shape (N, num_templates, C, H, W).
                Typically, there are 2 template images, and
                H and W are both equal to 128.

            img_metas (list[dict]): list of image information dict where each
                dict has: 'img_shape', 'scale_factor', 'flip', and may also
                contain 'filename', 'ori_shape', 'pad_shape', and
                'img_norm_cfg'. For details on the values of these keys see
                `mmtrack/datasets/pipelines/formatting.py:VideoCollect`.

            search_img (Tensor): of shape (N, 1, C, H, W) encoding input search
                images. 1 denotes there is only one search image for each
                template image. Typically H and W are both equal to 320.

            search_img_metas (list[list[dict]]): The second list only has one
                element. The first list contains search image information dict
                where each dict has: 'img_shape', 'scale_factor', 'flip', and
                may also contain 'filename', 'ori_shape', 'pad_shape', and
                'img_norm_cfg'. For details on the values of these keys see
                `mmtrack/datasets/pipelines/formatting.py:VideoCollect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for template
                images with shape (N, 4) in [tl_x, tl_y, br_x, br_y] format.

            padding_mask (Tensor): padding mask of template images.
                It's of shape (N, num_templates, H, W).
                Typically, there are 2 padding masks of template images, and
                H and W are both equal to that of template images.

            search_gt_bboxes (list[Tensor]): Ground truth bboxes for search
                images with shape (N, 5) in [0., tl_x, tl_y, br_x, br_y]
                format.

            search_padding_mask (Tensor): padding mask of search images.
                Its of shape (N, 1, H, W).
                There are 1 padding masks of search image, and
                H and W are both equal to that of search image.

            search_gt_labels (Tensor, optional): the categories of bbox:
                positive(1) or negative(0). It's of shape (N, 2). We just use
                search_gt_labels[:, 1].

        Returns:
            dict[str, Tensor]: a dictionary of loss components.
        """
        z1_dict = self.forward_before_head(img[:, 0], padding_mask[:, 0])
        z2_dict = self.forward_before_head(img[:, 1], padding_mask[:, 1])
        x_dict = self.forward_before_head(search_img[:, 0],
                                          search_padding_mask[:, 0])
        inputs = [z1_dict, z2_dict, x_dict]
        head_dict_inputs = self._merge_template_search(inputs)
        # run the transformer
        '''
        `track_results` is a dict containing the following keys:
            - 'pred_bboxes': bboxes of (N, num_query, 4) shape in
                    [tl_x, tl_y, br_x, br_y] format.
            - 'pred_logits': bboxes of (N, num_query, 1) shape.
        Typically `num_query` is equal to 1.
        '''
        track_results = self.head(head_dict_inputs)

        losses = dict()
        if self.head.run_bbox_head:
            img_size = search_img[:, 0].shape[2]
            tracking_bboxes = track_results['pred_bboxes'][:, 0] / img_size
            search_gt_bboxes = (
                torch.cat(search_gt_bboxes, dim=0).type(torch.float32)[:, 1:] /
                img_size).clamp(0., 1.)
            head_losses = self.head.reg_loss(tracking_bboxes, search_gt_bboxes)
        elif self.head.run_cls_head:
            assert search_gt_labels is not None
            pred_logits = track_results['pred_logits'][:, 0].squeeze()
            search_gt_labels = torch.cat(
                search_gt_labels, dim=0)[:, 1].squeeze()
            head_losses = self.head.cls_loss(pred_logits, search_gt_labels)

        losses.update(head_losses)

        return losses

    def simple_test(self, img, img_metas, **kwargs):
        pass
