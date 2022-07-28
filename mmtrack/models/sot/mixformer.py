# Copyright (c) OpenMMLab. All rights reserved.
import math
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from addict import Dict
from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh
from mmdet.models.builder import build_backbone, build_head, build_neck
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvNd
from torchvision.transforms.functional import normalize

from ..builder import MODELS
from .base import BaseSingleObjectTracker


@MODELS.register_module()
class MixFormer(BaseSingleObjectTracker):
    """MixFormer: End-to-End Tracking with Iterative Mixed Attention.

    This single object tracker is the implementation of `MixFormer<https://arxiv.org/abs/2203.11082>`.
    
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 init_cfg=None,
                 frozen_modules=None,
                 train_cfg=None,
                 test_cfg=None):
        super(MixFormer, self).__init__(init_cfg)
        self.backbone = build_backbone(backbone)
        # self.neck = build_neck(neck)
        self.head = build_head(head)
        print("Done")

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg

        # Set the update interval
        self.update_interval = self.test_cfg['update_interval']
        print("Update interval is: ", self.update_interval)
        self.online_size = self.test_cfg['online_size']
        print("Online size is: ", self.online_size)
        self.max_score_decay = 1.0
        print("Max score decay is: ", self.max_score_decay)

        # Set update interval, forget, online size

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
            raise NotImplementedError

        if self.with_head:
            self.head.init_weights()


    def set_online(self, template, online_template):
        self.backbone.set_online(template, online_template)

    def init(self, img, bbox):
        """Initialize the single object tracker in the first frame.
        
        Args:
            img: (Tensor): input image of shape (1, C, H, W).
            bbox (list | Tensor): in [cx, cy, w, h] format.
        """
        self.z_dict_list = []   # store templates
        # get the 1st template
        z_patch, _, z_mask = self.get_cropped_img(
            img, bbox, self.test_cfg["template_factor"],
            self.test_cfg["template_size"]
        )   # z_patch pf shape [1,C,H,W]; z_mask of shape [1,H,W]
        z_patch = normalize(
            z_patch.squeeze() / 255.,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ).unsqueeze(0)
        with torch.no_grad():
            self.set_online(z_patch, z_patch)

        self.template = z_patch
        self.online_template = z_patch
        self.best_online_template = z_patch
        self.best_conf_score = -1.0
        self.online_forget_id = 0

    def get_cropped_img(self, img, target_bbox, search_area_factor,
                        output_size):
        """ Crop Image
        Only used during testing
        This functino mainly contains two steps:
        1. Crop `img` based on target_bbox and search_area_factor.  If the 
        cropped image/mask is out of boundary of `img`, use 0 to pad.
        2. Resize the cropped image/mask to `output_size`.
        
        Args:
            img (Tensor): of shape (1, C, H, W)
            target_bbox (list | ndarray): in [cx, cy, w, h] format
            search_area_factor (float): ratio of crop size to target size
            output_size (float): the size of output cropped image
                (alwats square).
        Returns:
            img_crop_padded (Tensor): of shape (1, C, output_size, output_size)
            resize_factor (float): the ratio of original image scale to cropped
                image scale.
            padding_mask (Tensor): the padding mask caused by cropping. It's
                of shape (1, output_size, output_size)
        """
        cx, cy, w, h = target_bbox.split((1, 1, 1, 1), dim=-1)

        img_h, img_w = img.shape[2:]
        # 1. Crop image
        # 1.1 calculate crop size and pad size
        assert w > 0 and h > 0, f"Invalid bbox size ({w}, {h})!"
        crop_size = math.ceil(math.sqrt(w * h) * search_area_factor)
        if crop_size < 1:
            raise Exception("Too small bounding box.")

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
            value=0,
        )
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
            mode="bilinear",
            align_corners=False,
        )
        padding_mask = F.interpolate(
            padding_mask[None, None], (output_size, output_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(dim=0).type(torch.bool)

        return img_crop_padded, resize_factor, padding_mask

    def update_template(self, img, bbox, conf_score):
        """Update the dynamic templates.
        
        Args:
            img (Tensor): of shape (1, C, H, W).
            bbox (list | ndarray): in [cx, cy, w, h] format.
            conf_score (float): the confidence score of the predicted bbox.
        """
        if conf_score > 0.5 and conf_score > self.best_conf_score:
            z_patch, _, z_mask = self.get_cropped_img(
                img,
                bbox,
                self.test_cfg["template_factor"],
                output_size=self.test_cfg["template_size"],
            )
            z_patch = normalize(
                z_patch.squeeze() / 255.,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ).unsqueeze(0)
            self.best_online_template = z_patch
            self.best_conf_score = conf_score
        if self.frame_id % self.update_interval == 0:
            if self.online_size == 1:
                self.online_template = self.best_online_template
            elif self.online_template.shape[0] < self.online_size:
                self.online_template = torch.cat([self.online_template, self.best_online_template])
            else:
                self.online_template[self.online_forget_id:self.online_forget_id+1] = self.best_online_template
                self.online_forget_id = (self.online_forget_id + 1) % self.online_size

            if self.online_size > 1:
                with torch.no_grad():
                    self.set_online(self.template, self.online_template)

            self.best_conf_score = -1
            self.best_online_template = self.template

    def track(self, img, bbox):
        """Track the box `bbox` of previous frame to current frame `img`

        Args:
            img (Tensor): of shape (1, C, H, W).
            bbox (list | Tensor): The bbox in previous frame. The shape of the 
                bbox is (4, ) in [x, y, w, h] format.
        """
        H, W = img.shape[2:]
        # get the t-th search region
        x_patch, resize_factor, x_mask = self.get_cropped_img(
            img, bbox, self.test_cfg["search_factor"],
            self.test_cfg["search_size"]
        )   # bbox: of shape (x1,y1, w, h), x_mask: of shape (1, h, w)
        x_patch = normalize(
            x_patch.squeeze() / 255.,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ).unsqueeze(0)

        with torch.no_grad():
            x_patch.squeeze(1)
            template, search = self.backbone.forward_test(x_patch)
            out_dict = self.head(template, search)

        pred_box = out_dict['pred_bboxes']    # xyxy
        # Baseline: Take the mean of all pred boxes as the final result

        pred_box = self.mapping_bbox_back(pred_box, self.memo.bbox, resize_factor)
        pred_box = self._bbox_clip(pred_box, H, W, margin=10)

        # update template
        conf_score = -1.
        if self.head.score_head is not None:
            # get confidence score (whether the search region is reliable)
            conf_score = out_dict['pred_scores'].view(1).sigmoid().item()
            crop_bbox = bbox_xyxy_to_cxcywh(pred_box)
            self.update_template(img, crop_bbox, conf_score)

        return conf_score, pred_box

    def mapping_bbox_back(self, pred_bboxes, prev_bbox, resize_factor):
        """Mapping the `prediction bboxes` from resized cropped image to
        original image. The coordinate origins of them are both the top left
        corner.
        
        Args:
            pred_bboxes (Tensor): the predicted bboxes of shape (B, Nq, 4), in 
                [tl_x, tl_y, br_x, br_y] format. The coordinates are based in 
                the resized cropped image.
            prev_bbox (Tensor): the previous bbox of shape (B, 4),
                in [cx, cy, w, h] format. The coordinates are based in the 
                original image.
            resize_factor (float): the ratio of original image scale to cropped 
                image scale.
        Returns:
            (Tensor): in [tl_x, tl_y, br_x, br_y] format.
        """
        # based in the resized croped image
        pred_bboxes = pred_bboxes.view(-1, 4)
        # based in the original croped image
        pred_bbox = pred_bboxes.mean(dim=0) / resize_factor

        # the half size of the original croped image
        cropped_img_half_size = 0.5 * self.test_cfg["search_size"] / resize_factor
        # (x_shift, y_shift) is the coordinate of top left corner of the
        # cropped image basesd in the original image.
        x_shift, y_shift = prev_bbox[0] - cropped_img_half_size, prev_bbox[1] - cropped_img_half_size
        pred_bbox[0:4:2] += x_shift
        pred_bbox[1:4:2] += y_shift

        return pred_bbox

    def _bbox_clip(self, bbox, img_h, img_w, margin=0):
        """Clip the bboxin [tl_x, tl_y, br_x, br_y] format."""
        bbox_w, bbox_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        bbox[0] = bbox[0].clamp(0, img_w - margin)
        bbox[1] = bbox[1].clamp(0, img_h - margin)
        bbox[2] = bbox[0] + bbox_w
        bbox[3] = bbox[1] + bbox_h
        return bbox

    def simple_test(self, img, img_metas, gt_bboxes, **kwargs):
        """Test without augmentation.

        Args:
            img (Tensor): input image of shape (1, C, H, W).
            img_metas (list[dict]): list of image information dict where each
                dict has: 'img_shape', 'scale_factor', 'flip', and may also 
                contain 'filename', 'ori_shape', 'pad_shape', and 
                'img_norm_cfg'. For details on the values of these keys see
                `mmtrack/datasets/pipelines/formatting.py:VideoColloet`.
            gt_bboxes (list[Tensor]): list of ground truth bboxes for each 
                image with shape (1, 4) in [tl_x, tl_y, br_x, br_y] format.

        Returns:
            dict(str: ndarray): the tracking results.
        """
        frame_id = img_metas[0].get('frame_id', -1)
        assert frame_id >= 0
        assert len(img) == 1, "only support batch_size=1 when testing"
        self.frame_id = frame_id

        if frame_id == 0:
            bbox_pred = gt_bboxes[0][0]
            self.memo = Dict()
            self.memo.bbox = bbox_xyxy_to_cxcywh(bbox_pred)
            self.init(img, self.memo.bbox)
            best_score = -1.
        else:
            best_score, bbox_pred = self.track(img, self.memo.bbox)
            self.memo.bbox = bbox_xyxy_to_cxcywh(bbox_pred)

        results = dict()
        results["track_bboxes"] = np.concatenate(
            (bbox_pred.cpu().numpy(), np.array([best_score]))
        )
        return results

    def forward_train(self,
                      imgs,
                      img_metas,
                      search_img,
                      search_img_metas,
                      **kwargs):
        """forward of training

        Args:
            img (Tensor): template images of shape (N, num_templates, C, H, W)
                Typically, there are 2 template images, and H and W are both equal to 128.

            img_metas (list[dict]): list of image information dict where each
                dict has: 'image_shape', 'scale_factor', 'flip', and may also
                contain 'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'. For details on the values of these keys see
                `mmtrack/datasets/pipelines/formatting.py:VideoCollect`.

            search_img (Tensor): of shape (N, 1, C, H, W) encoding input search
                images. 1 denotes there is only one search image for each
                exemplar image. Typically H and W are both equal to 320.

            search_img_metas (list[list[dict]]): The second list only has one
                element. The first list contains search image information dict
                where each dict has: 'img_shape', 'scale_factor', 'flip', and
                may also contain 'filename', 'ori_shape', 'pad_shape' and
                'img_norm_cfg'. For details on the values of there keys see
                `mmtrack/datasets/pipelines/formatting.py:VideoCollect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for template
                images with shape (N, 4) in [tl_x, tl_y, br_x, br_y] format.

            padding_mask (Tensor): padding mask of template images.
                It's of shape (N, num_templates, H, W).
                Typically, there are 2 padding masks of tehmplate images, and
                H and W are both equal to that of template images.

            search_gt_bboxes (list[Tensor]): Ground truth bboxes for search
                images with shape (N, 5) in [0., tl_x, tl_y, br_x, br_y] format.

            search_padding_mask (Tensor): padding mask of search images.
                Its of shape (N, 1, H, W).
                There are 1 padding masks of search image, and
                H and W are both equal to that of search image.

            search_gt_labels (list[Tensor], optional): Ground truth labels for
                search images with shape (N, 2).
        
        Returns:
            dict[str, Tensor]: a dictionary of loss components.
        """
        template, online_template = imgs[:, 0, ...], imgs[:, 1, ...]
        search = search_img.squeeze(1)
        template, search = self.backbone(template, online_template, search)

        # box head
        out_dict = self.head(template, search, **kwargs)

        # compute loss
        return out_dict
     