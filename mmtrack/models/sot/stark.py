# Copyright (c) OpenMMLab. All rights reserved.
import math
from collections import defaultdict
from copy import deepcopy

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from addict import Dict
from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh
from mmdet.models.builder import build_backbone, build_head, build_neck
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvNd

from ..builder import MODELS
from .base import BaseSingleObjectTracker


@MODELS.register_module()
class Stark(BaseSingleObjectTracker):
    """"""

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

    def get_cropped_img(self, img, target_bbox, search_area_factor,
                        output_size):
        """ Crop Image
        Only used during testing
        This function mainly contains two steps:
        1. Crop `img` based on target_bbox and search_area_factor. If the
        cropped image/mask is out of boundary of `img`, use 0 to pad.
        2. Resize the cropped image/mask to `output_size`.

        args:
            img (Tensor): of shape (B, C, H, W)
            target_bbox (list | ndarray | array): in [cx, cy, w, h] format
            search_area_factor (float): Ratio of crop size to target size
            output_size (float): Size to which the extracted crop is resized
            (always square).
        returns:
            img_crop_padded (Tensor): of shape (B, C, H, W)
            resize_factor (float): the factor by which the crop has been
            resized to make the
                crop size equal output_size
            att_mask (ndarray)
        """
        cx, cy, w, h = target_bbox.split((1, 1, 1, 1), dim=-1)

        _, _, img_h, img_w = img.shape
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
        # 1.4 pad attention mask
        _, _, img_h, img_w = img_crop_padded.shape
        att_mask = np.ones((img_h, img_w))
        end_x, end_y = -x2_pad, -y2_pad
        if y2_pad == 0:
            end_y = None
        if x2_pad == 0:
            end_x = None
        att_mask[y1_pad:end_y, x1_pad:end_x] = 0

        # 2. Resize image and attention mask
        resize_factor = output_size / crop_size
        img_crop_padded_numpy = np.transpose(
            img_crop_padded.squeeze().cpu().numpy(), (1, 2, 0))
        img_crop_padded_numpy = cv2.resize(img_crop_padded_numpy,
                                           (output_size, output_size))
        img_crop_padded = torch.from_numpy(img_crop_padded_numpy).permute(
            (2, 0, 1)).unsqueeze(0).to(img.device)

        att_mask = cv2.resize(att_mask,
                              (output_size, output_size)).astype(np.bool_)

        return img_crop_padded, resize_factor, att_mask

    def _normalize(self, img_tensor, amask_arr):
        self.mean = torch.tensor([0.485, 0.456,
                                  0.406]).to(img_tensor.device)[None, :, None,
                                                                None]
        self.std = torch.tensor([0.229, 0.224,
                                 0.225]).to(img_tensor.device)[None, :, None,
                                                               None]

        # Deal with the image patch
        img_tensor_norm = (
            (img_tensor / 255.0) - self.mean) / self.std  # (1,3,H,W)
        # Deal with the attention mask
        amask_tensor = torch.from_numpy(amask_arr).to(
            torch.bool).unsqueeze(dim=0).to(img_tensor.device)  # (1,H,W)
        return img_tensor_norm, amask_tensor

    def _merge_template_search(self, inputs):
        """NOTICE: search region related features must be in the last place.
        args:
            inputs: list(dict)
                - 'feat': (bs, c, h, w)
                - 'mask': (bs, h, w)
                - 'pos_embed': (bs, c, h ,w)

        The merge includes 3 steps: flatten, premute and concatenate.

        Return:
            seq_dict: (dict)
            - 'feat': [z1_h*z1_w + z2_h*z2_w + x_h*x_w, bs, c]
            - 'mask': [bs, z1_h*z1_w + z2_h*z2_w + x_h*x_w]
            - 'pos_embed': [z1_h*z1_w + z2_h*z2_w + x_h*x_w, bs, c]
        """
        seq_dict = defaultdict(list)
        for input_dic in inputs:
            for name, x in input_dic.items():
                if name == 'mask':
                    seq_dict[name].append(x.flatten(1))
                else:
                    seq_dict[name].append(x.flatten(2).permute(2, 0, 1))
        for name, x in seq_dict.items():
            if name == 'mask':
                seq_dict[name] = torch.cat(x, dim=1)
            else:
                seq_dict[name] = torch.cat(x, dim=0)
        return seq_dict

    def forward_before_head(self, img, mask):
        """Extract the features of exemplar images.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input exemplar
                images. Typically H and W equal to 127.

        Returns:
            tuple(Tensor): Multi level feature map of exemplar images.
        """
        feat = self.backbone(img)
        feat = self.neck(feat)[0]

        mask = F.interpolate(
            mask[None].float(), size=feat.shape[-2:]).to(torch.bool)[0]
        pos_embed = self.head.positional_encoding(mask)

        return {'feat': feat, 'mask': mask, 'pos_embed': pos_embed}

    def init(self, img, bbox):
        """Initialize the single object tracker in the first frame.

        Args:
            img (Tensor): of shape (1, C, H, W) encoding original input
                image.
            bbox (List | Tensor): [cx, cy, w, h]
        """
        # initialize z_dict_list
        self.z_dict_list = []
        # get the 1st template
        z_patch, _, z_mask = self.get_cropped_img(
            img.type(torch.uint8), bbox, self.test_cfg['template_factor'],
            self.test_cfg['template_size'])
        # z_patch:(1,C,H,W);  z_mask:(1,H,W)
        z_patch, z_mask = self._normalize(z_patch, z_mask)
        with torch.no_grad():
            self.z_dict = self.forward_before_head(z_patch, z_mask)
        self.z_dict_list.append(self.z_dict)

        # get the complete z_dict_list
        for _ in range(self.num_extra_template):
            self.z_dict_list.append(deepcopy(self.z_dict))

    def update_template(self, img, bbox, conf_score):
        for idx, update_i in enumerate(self.update_intervals):
            if self.frame_id % update_i == 0 and conf_score > 0.5:
                z_patch, _, z_mask = self.get_cropped_img(
                    img.type(torch.uint8),
                    bbox,
                    self.test_cfg['template_factor'],
                    output_size=self.test_cfg['template_size'])
                z_patch, z_mask = self._normalize(z_patch, z_mask)
                with torch.no_grad():
                    z_dict_dymanic = self.forward_before_head(z_patch, z_mask)
                # the 1st element of z_dict_list is template from the 1st frame
                self.z_dict_list[idx + 1] = z_dict_dymanic

    def mapping_bbox_back(self, pred_bboxes, prev_bbox, resize_factor):
        """This function map the `prediction bboxes` from resized croped image
        to original image. The coordinate origins of them are both the top left
        corner.

        Args:
            pred_bboxes (Tensor): of shape (B, Nq, 4), in
            [tl_x, tl_y, br_x, br_y] format.
            prev_bbox (Tensor): of shape (B, 4), in [cx, cy, w, h] format.
            resize_factor (float):pred_bboxes
        Returns:
            (Tensor): in [tl_x, tl_y, br_x, br_y] format
        """
        # based in resized croped image
        pred_bboxes = pred_bboxes.view(-1, 4)
        # based in original croped image
        pred_bbox = pred_bboxes.mean(dim=0) / resize_factor  # (cx, cy, w, h)

        # map the bbox to original image
        half_crop_img_size = 0.5 * self.test_cfg['search_size'] / resize_factor
        x_shift, y_shift = prev_bbox[0] - half_crop_img_size, prev_bbox[
            1] - half_crop_img_size
        pred_bbox[0] += x_shift
        pred_bbox[1] += y_shift
        pred_bbox[2] += x_shift
        pred_bbox[3] += y_shift

        return pred_bbox

    def track(self, img, bbox):
        """Track the box `bbox` of previous frame to current frame `img`.

        Args:
            img (Tensor): of shape (1, C, H, W) encoding original input
                image.
            bbox (List or Tensor): The bbox in previous frame. The shape of the
                bbox is (4, ) in [x, y, w, h] format.

        Returns:
        """
        H, W = img.shape[2:]
        # get the t-th search region
        x_patch, resize_factor, x_mask = self.get_cropped_img(
            img.type(torch.uint8), bbox, self.test_cfg['search_factor'],
            self.test_cfg['search_size'])  # bbox (x1, y1, w, h)
        # x_mask:(1,h,w)
        x_patch, x_mask = self._normalize(x_patch, x_mask)

        with torch.no_grad():
            x_dict = self.forward_before_head(x_patch, x_mask)
            head_dict_inputs = self._merge_template_search(self.z_dict_list +
                                                           [x_dict])
            # run the transformer
            track_results, _ = self.head(
                head_dict_inputs, run_box_head=True, run_cls_head=False)

        # get confidence score (whether the search region is reliable)
        conf_score = track_results['pred_logits'].view(-1).sigmoid().item()

        final_bbox = self.mapping_bbox_back(track_results['pred_bboxes'],
                                            self.memo.bbox, resize_factor)
        final_bbox = self._bbox_clip(final_bbox, H, W, margin=10)
        self.update_template(img, final_bbox, conf_score)

        return conf_score, final_bbox

    def _bbox_clip(self, bbox, img_h, img_w, margin=0):
        """Clip the bbox with [tl_x, tl_y, br_x, br_y] format."""
        bbox_w, bbox_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        bbox[0] = bbox[0].clamp(0, img_w - margin)
        bbox[1] = bbox[1].clamp(0, img_h - margin)
        bbox_w = bbox_w.clamp(margin, img_w)
        bbox_h = bbox_h.clamp(margin, img_h)
        bbox[2] = bbox[0] + bbox_w
        bbox[3] = bbox[1] + bbox_h
        return bbox

    def simple_test(self, img, img_metas, gt_bboxes, **kwargs):
        """Test without augmentation.

        Args:
            img (Tensor): of shape (1, C, H, W) encoding input image.
            img_metas (list[dict]): list of image information dict where each
                dict has: 'img_shape', 'scale_factor', 'flip', and may also
                contain 'filename', 'ori_shape', 'pad_shape', and
                'img_norm_cfg'. For details on the values of these keys see
                `mmtrack/datasets/pipelines/formatting.py:VideoCollect`.
            gt_bboxes (list[Tensor]): list of ground truth bboxes for each
                image with shape (1, 4) in [tl_x, tl_y, br_x, br_y] format.

        Returns:
            dict[str : ndarray]: The tracking results.
        """
        frame_id = img_metas[0].get('frame_id', -1)
        assert frame_id >= 0
        assert len(img) == 1, 'only support batch_size=1 when testing'
        self.frame_id = frame_id

        if frame_id == 0:
            bbox_pred = gt_bboxes[0][0]
            self.memo = Dict()
            self.memo.bbox = bbox_xyxy_to_cxcywh(bbox_pred)
            # self.memo.bbox = self.bbox_xyxy_to_xywh(gt_bboxes)  # [x,y,w,h]
            # get the templates stored in self.z_dict_list
            self.init(img, self.memo.bbox)
            best_score = -1.
        else:
            best_score, bbox_pred = self.track(img, self.memo.bbox)
            self.memo.bbox = bbox_xyxy_to_cxcywh(bbox_pred)

        # tl_x, tl_y, br_x, br_y = bbox_pred.cpu().tolist()
        # print(best_score, [tl_x, tl_y, br_x-tl_x, br_y-tl_y])

        results = dict()
        results['track_bboxes'] = np.concatenate(
            (bbox_pred.cpu().numpy(), np.array([best_score])))
        return results

    def forward_train(self, img, img_metas, gt_bboxes, search_img, att_mask,
                      search_img_metas, search_gt_bboxes, search_att_mask,
                      **kwargs):

        z1_dict = self.forward_before_head(img[:, 0], att_mask[:, 0])
        z2_dict = self.forward_before_head(img[:, 1], att_mask[:, 1])
        x_dict = self.forward_before_head(search_img[:, 0], search_att_mask[:,
                                                                            0])
        inputs = [z1_dict, z2_dict, x_dict]
        head_dict_inputs = self._merge_template_search(inputs)
        # run the transformer
        track_results, _ = self.head(
            head_dict_inputs, run_box_head=True, run_cls_head=False)

        img_size = search_img[:, 0].shape[2]
        tracking_bboxes = track_results['pred_bboxes'][:, 0] / img_size
        search_gt_bboxes = (
            torch.cat(search_gt_bboxes, dim=0).type(torch.float32)[:, 1:] /
            img_size).clamp(0., 1.)
        # print(tracking_bboxes, search_gt_bboxes)
        losses = dict()
        if self.head.run_bbox_head:
            head_losses = self.head.reg_loss(tracking_bboxes, search_gt_bboxes)
        losses.update(head_losses)

        return losses
