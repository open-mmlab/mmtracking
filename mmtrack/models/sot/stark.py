# Copyright (c) OpenMMLab. All rights reserved.
import math
import os.path as osp
import warnings
from collections import defaultdict
from copy import deepcopy

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from addict import Dict
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
                 pretrains=None,
                 init_cfg=None,
                 frozen_modules=None,
                 train_cfg=None,
                 test_cfg=None):
        super(Stark, self).__init__(init_cfg)
        if isinstance(pretrains, dict):
            warnings.warn('DeprecationWarning: pretrains is deprecated, '
                          'please use "init_cfg" instead')
            backbone_pretrain = pretrains.get('backbone', None)
            if backbone_pretrain:
                backbone.init_cfg = dict(
                    type='Pretrained', checkpoint=backbone_pretrain)
            else:
                backbone.init_cfg = None
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        head.update(test_cfg=test_cfg)
        self.head = build_head(head)

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg

        # Set the update interval
        self.update_intervals = self.test_cfg['update_intervals']
        print('Update interval is: ', self.update_intervals)
        self.num_extra_template = len(self.update_intervals)

        self.save_all_boxes = False
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
            for m in self.head.modules():
                if isinstance(m, _ConvNd) or isinstance(m, _BatchNorm):
                    m.reset_parameters()

    def sample_target(self,
                      im,
                      target_bb,
                      search_area_factor,
                      output_sz=None,
                      mask=None):
        """ Extracts a square crop centered at target_bb box, of area
        search_area_factor^2 times target_bb area
        args:
            im - cv image
            target_bb - target box [x, y, w, h]
            search_area_factor - Ratio of crop size to target size
            output_sz - (float) Size to which the extracted crop is resized
            (always square). If None, no resizing is done.
        returns:
            cv image - extracted crop
            float - the factor by which the crop has been resized to make the
            crop size equal output_size
        """

        if not isinstance(target_bb, list):
            x, y, w, h = target_bb.tolist()
        else:
            x, y, w, h = target_bb
        _, _, H, W = im.shape
        # Crop image
        crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)

        if crop_sz < 1:
            raise Exception('Too small bounding box.')

        x1 = round(x + 0.5 * w - crop_sz * 0.5)
        x2 = x1 + crop_sz

        y1 = round(y + 0.5 * h - crop_sz * 0.5)
        y2 = y1 + crop_sz

        x1_pad = max(0, -x1)
        x2_pad = max(x2 - W + 1, 0)

        y1_pad = max(0, -y1)
        y2_pad = max(y2 - H + 1, 0)

        # Crop target
        # im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]
        im_crop = im[..., y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad]
        if mask is not None:
            mask_crop = mask[..., y1 + y1_pad:y2 - y2_pad,
                             x1 + x1_pad:x2 - x2_pad]

        # Pad
        # im_crop_padded = cv2.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad,
        #                                     x2_pad, cv2.BORDER_CONSTANT)
        im_crop_padded = F.pad(
            im_crop,
            pad=(x1_pad, x2_pad, y1_pad, y2_pad),
            mode='constant',
            value=0)
        # deal with attention mask
        bs, _, H, W = im_crop_padded.shape
        att_mask = np.ones((H, W))
        end_x, end_y = -x2_pad, -y2_pad
        if y2_pad == 0:
            end_y = None
        if x2_pad == 0:
            end_x = None
        att_mask[y1_pad:end_y, x1_pad:end_x] = 0
        if mask is not None:
            mask_crop_padded = F.pad(
                mask_crop,
                pad=(x1_pad, x2_pad, y1_pad, y2_pad),
                mode='constant',
                value=0)

        if output_sz is not None:
            resize_factor = output_sz / crop_sz
            # im_crop_padded = cv2.resize(im_crop_padded,
            # (output_sz, output_sz))
            im_crop_padded_numpy = np.transpose(im_crop_padded.squeeze().cpu().numpy(),(1,2,0))
            im_crop_padded_numpy = cv2.resize(im_crop_padded_numpy,(output_sz, output_sz))
            im_crop_padded = torch.from_numpy(im_crop_padded_numpy).permute((2,0,1)).unsqueeze(0).to(im.device)


            # im_crop_padded = F.interpolate(
            #     im_crop_padded, (output_sz, output_sz),
            #     mode='bilinear',
            #     align_corners=False)
            # TODO use F.interpolate
            att_mask = cv2.resize(att_mask,
                                  (output_sz, output_sz)).astype(np.bool_)

            if mask is None:
                return im_crop_padded, resize_factor, att_mask
            mask_crop_padded = F.interpolate(
                mask_crop_padded[None, None], (output_sz, output_sz),
                mode='bilinear',
                align_corners=False)[0, 0]
            return im_crop_padded, resize_factor, att_mask, mask_crop_padded

        else:
            if mask is None:
                return im_crop_padded, att_mask.astype(np.bool_), 1.0
            return im_crop_padded, 1.0, att_mask.astype(
                np.bool_), mask_crop_padded

    def _normalize(self, img_tensor, amask_arr):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(
            (1, 3, 1, 1)).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(
            (1, 3, 1, 1)).cuda()

        # Deal with the image patch
        # img_tensor = torch.tensor(img_arr).cuda().float().permute(
        #     (2, 0, 1)).unsqueeze(dim=0)
        img_tensor_norm = (
            (img_tensor / 255.0) - self.mean) / self.std  # (1,3,H,W)
        # Deal with the attention mask
        amask_tensor = torch.from_numpy(amask_arr).to(
            torch.bool).cuda().unsqueeze(dim=0)  # (1,H,W)
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
        with torch.no_grad():
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
            bbox (List or Tensor): [x, y, w, h]

        Returns:
            tuple(z_feat, avg_channel): z_feat is a tuple[Tensor] that
            contains the multi level feature maps of exemplar image,
            avg_channel is Tensor with shape (3, ), and denotes the padding
            values.
        """
        # initialize z_dict_list
        self.z_dict_list = []
        # get the 1st template
        z_patch, _, z_mask = self.sample_target(
            img.type(torch.uint8),
            bbox,
            self.test_cfg['template_factor'],
            output_sz=self.test_cfg['template_size'])
        # z_patch:(1,C,H,W);  z_mask:(1,H,W)
        z_patch, z_mask = self._normalize(z_patch, z_mask)
        self.z_dict = self.forward_before_head(z_patch, z_mask)
        self.z_dict_list.append(self.z_dict)

        # get the complete z_dict_list
        for _ in range(self.num_extra_template):
            self.z_dict_list.append(deepcopy(self.z_dict))

        if self.save_all_boxes:
            """save all predicted boxes."""
            all_boxes_save = bbox * self.cfg.num_object_queries
            return {'all_boxes': all_boxes_save}

    def _clip_box(self, box: list, H, W, margin=0):
        x1, y1, w, h = box
        x2, y2 = x1 + w, y1 + h
        x1 = min(max(0, x1), W - margin)
        x2 = min(max(margin, x2), W)
        y1 = min(max(0, y1), H - margin)
        y2 = min(max(margin, y2), H)
        w = max(margin, x2 - x1)
        h = max(margin, y2 - y1)
        return [x1, y1, w, h]

    def track(self, img, bbox):
        """Track the box `bbox` of previous frame to current frame `img`.

        Args:
            img (Tensor): of shape (1, C, H, W) encoding original input
                image.
            bbox (List or Tensor): The bbox in previous frame. The shape of the
            box is
                (4, ) in [x, y, w, h] format.

        Returns:
        """
        H, W = img.shape[2:]
        # get the t-th search region
        x_patch, resize_factor, x_mask = self.sample_target(
            img.type(torch.uint8),
            bbox,
            self.test_cfg['search_factor'],
            output_sz=self.test_cfg['search_size'])  # bbox (x1, y1, w, h)
        # x_mask:(1,h,w)
        x_patch, x_mask = self._normalize(x_patch, x_mask)

        with torch.no_grad():
            x_dict = self.forward_before_head(x_patch, x_mask)
            head_dict_inputs = self._merge_template_search(self.z_dict_list +
                                                           [x_dict])
            # run the transformer
            track_results, _ = self.head(
                head_dict_inputs, run_box_head=True, run_cls_head=True)

        # get the final result
        pred_boxes = track_results['pred_boxes'].view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * self.test_cfg['search_size'] /
                    resize_factor)  # (cx, cy, w, h) [0,1]
        pred_box = pred_box.tolist()
        # get the final box result
        bbox = self._clip_box(
            self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        # get confidence score (whether the search region is reliable)
        conf_score = track_results['pred_logits'].view(-1).sigmoid().item()
        # update template
        for idx, update_i in enumerate(self.update_intervals):
            if self.frame_id % update_i == 0 and conf_score > 0.5:
                z_patch, _, z_mask = self.sample_target(
                    img.type(torch.uint8),
                    bbox,
                    self.test_cfg['template_factor'],
                    output_sz=self.test_cfg['template_size'])  # (x1, y1, w, h)
                z_patch, z_mask = self._normalize(z_patch, z_mask)
                with torch.no_grad():
                    z_dict_dymanic = self.forward_before_head(z_patch, z_mask)
                # the 1st element of z_dict_list is template from the 1st frame
                self.z_dict_list[idx + 1] = z_dict_dymanic
        print(bbox, conf_score)

        # for debug
        if self.debug:
            x1, y1, w, h = bbox
            image_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.rectangle(
                image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)),
                color=(0, 0, 255),
                thickness=2)
            save_path = osp.join(self.save_dir, '%04d.jpg' % self.frame_id)
            cv2.imwrite(save_path, image_BGR)
        if self.save_all_boxes:
            """save all predictions."""
            all_boxes = self.map_box_back_batch(
                pred_boxes * self.test_cfg['search_size'] / resize_factor,
                resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {
                'target_bbox': bbox,
                'all_boxes': all_boxes_save,
                'conf_score': conf_score
            }
        else:
            return {'target_bbox': bbox, 'conf_score': conf_score}

    def map_box_back(self, pred_box: list, resize_factor: float):
        '''
        pred_bbox: [cx,cy,w,h]
        return: [x,y,w,h]
        '''
        cx_prev, cy_prev = self.memo.bbox[0] + 0.5 * self.memo.bbox[
            2], self.memo.bbox[1] + 0.5 * self.memo.bbox[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.test_cfg['search_size'] / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.memo.bbox[0] + 0.5 * self.memo.bbox[
            2], self.memo.bbox[1] + 0.5 * self.memo.bbox[3]
        cx, cy, w, h = pred_box.unbind(-1)  # (N,4) --> (N,)
        half_side = 0.5 * self.test_cfg['search_size'] / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h],
                           dim=-1)

    def bbox_xyxy_to_xywh(self, bbox):

        bbox[2] = max(bbox[2] - bbox[0], 0)
        bbox[3] = max(bbox[3] - bbox[1], 0)

        return bbox

    def bbox_xywh_to_xyxy(self, bbox, img_shape):

        bbox[2] = min(bbox[2] + bbox[0], img_shape[1])
        bbox[3] = min(bbox[3] + bbox[1], img_shape[0])

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
            gt_bboxes = gt_bboxes[0][0].tolist()
            self.memo = Dict()
            # self.memo.bbox = bbox_xyxy_to_cxcywh(gt_bboxes)
            self.memo.bbox = self.bbox_xyxy_to_xywh(gt_bboxes)  # [x,y,w,h]
            # get the templates stored in self.z_dict_list
            self.init(img, self.memo.bbox)
            best_score = -1.
        else:
            out_dict = self.track(img, self.memo.bbox)
            best_score, self.memo.bbox = out_dict['conf_score'], out_dict[
                'target_bbox']

        bbox_pred = self.bbox_xywh_to_xyxy(deepcopy(self.memo.bbox), img.shape[2:])
        results = dict()
        results['track_bboxes'] = np.array(bbox_pred + [best_score])

        return results

    def forward_train(self, ):
        pass
