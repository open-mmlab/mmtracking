# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from mmcv.image import imrotate
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from mmengine.structures import InstanceData
from torch import Tensor

from mmtrack.registry import MODELS
from mmtrack.utils import (InstanceList, OptConfigType, OptMultiConfig,
                           SampleList, gauss_blur)
from .base import BaseSingleObjectTracker


@MODELS.register_module()
class PrDiMP(BaseSingleObjectTracker):
    """PrDiMP: Probabilistic Regression for Visual Tracking.

    This single object tracker is the implementation of `PrDiMP
    <https://arxiv.org/abs/2003.12565>`_.

    args:
        backbone (dict, optional): the configuration of backbone network.
            Defaults to None.
        cls_head (dict, optional):  target classification module.
            Defaults to None.
        bbox_head (dict, optional):  bounding box regression module.
            Defaults to None.
        init_cfg (dict, optional): the configuration of initialization.
            Defaults to None.
        test_cfg (dict, optional): the configuration of test.
            Defaults to None.

    """

    def __init__(self,
                 backbone: dict,
                 neck: Optional[dict] = None,
                 cls_head: Optional[dict] = None,
                 bbox_head: Optional[dict] = None,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super(PrDiMP, self).__init__(data_preprocessor, init_cfg)
        self.backbone = MODELS.build(backbone)
        cls_head.update(test_cfg=test_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.classifier = MODELS.build(cls_head)
        self.bbox_regressor = MODELS.build(bbox_head)
        self.test_cfg = test_cfg

    def init(self, img: Tensor) -> Tuple[Tuple[Tensor, ...], Tensor]:
        """Initialize tracker.

        Args:
            img (Tensor): Input image of shape (1, C, H, W).
            init_bbox (Tensor): of (4, ) shape in [cx, cy, w, h] format.
        """
        self.frame_num = 1
        init_bbox = self.memo.bbox

        # Set size for image and cropped sample. img_size is in [w, h] format.
        self.img_size = torch.Tensor([img.shape[-1],
                                      img.shape[-2]]).to(init_bbox.device)
        sample_size = self.test_cfg['img_sample_size']
        sample_size = torch.Tensor([sample_size, sample_size] if isinstance(
            sample_size, int) else sample_size)
        self.sample_size = sample_size.to(init_bbox.device)

        # Compute expanded size and output size about augmentation
        aug_expansion_factor = self.test_cfg['init_aug_cfg'][
            'aug_expansion_factor']
        aug_expansion_size = (self.sample_size * aug_expansion_factor).float()

        # Crop image patches and bboxes
        img_patch, patch_coord = self.get_cropped_img(
            img, init_bbox.round(),
            self.test_cfg['search_scale_factor'] * aug_expansion_factor,
            aug_expansion_size)
        resized_factor = (patch_coord[:, 2:4] /
                          aug_expansion_size).prod().sqrt()
        init_bbox = self.generate_bbox(init_bbox, init_bbox[:2].round(),
                                       resized_factor)

        # Crop patches from the image and perform augmentation on the image
        # patches
        aug_img_patches, aug_cls_bboxes = self.gen_aug_imgs_bboxes(
            img_patch, init_bbox, self.sample_size)

        # `init_backbone_feats` is a tuple containing the features of `layer2`
        # and `layer3`
        init_backbone_feats = self.backbone(aug_img_patches)

        # Initialize the classifier with bboxes and features of `layer3`
        # get the augmented bboxes on the augmented image patches
        self.classifier.init_classifier(
            init_backbone_feats[-1], aug_cls_bboxes,
            self.test_cfg['init_aug_cfg']['augmentation']['dropout'])

        # Initialize IoUNet
        # only use the features of the non-augmented image
        init_iou_features = [x[:1] for x in init_backbone_feats]
        self.bbox_regressor.init_iou_net(init_iou_features, init_bbox)

    def img_shift_crop(self,
                       img: Tensor,
                       output_size: Optional[List] = None,
                       shift: Optional[List] = None) -> Tensor:
        """Shift and crop the image.

        Args:
            img (Tensor): The image of shape (C, H, W).
            output_size (list): in [w, h] format.
            shift (list): in [x, y] fotmat.

        Returns:
            Tensor: Augmented image.
        """
        img_size = [img.shape[-1], img.shape[-2]]
        # img_size = img.shape[-2:]
        if output_size is None:
            pad_h = 0
            pad_w = 0
        else:
            pad_w = (output_size[0] - img_size[0]) / 2
            pad_h = (output_size[1] - img_size[1]) / 2

        if shift is None:
            shift = [0, 0]

        pad_left = math.floor(pad_w) + shift[0]
        pad_right = math.ceil(pad_w) - shift[0]
        pad_top = math.floor(pad_h) + shift[1]
        pad_bottom = math.ceil(pad_h) - shift[1]

        return F.pad(img, (pad_left, pad_right, pad_top, pad_bottom),
                     'replicate')

    def gen_aug_imgs_bboxes(self, img: Tensor, init_bbox: Tensor,
                            output_size: Tensor) -> Tensor:
        """Perform data augmentation.

        Args:
            img (Tensor): Cropped image of shape (1, C, H, W).
            init_bbox (Tensor): of (4, ) shape in [cx, cy, w, h] format.
            output_size (Tensor): of (2, ) shape in [w, h] format.

        Returns:
            Tensor: The cropped augmented image patches.
        """
        output_size = output_size.long().cpu().tolist()
        random_shift_factor = self.test_cfg['init_aug_cfg'][
            'random_shift_factor']

        def get_rand_shift():
            return ((torch.rand(2) - 0.5) * self.sample_size.cpu() *
                    random_shift_factor).long().tolist()

        augs = self.test_cfg['init_aug_cfg']['augmentation']
        aug_imgs = [self.img_shift_crop(img, output_size)]
        aug_bboxes = [init_bbox]

        # All augmentations
        if 'relativeshift' in augs:
            for shift in augs['relativeshift']:
                absulute_shift = (torch.Tensor(shift) *
                                  self.sample_size.cpu() / 2).long().tolist()
                aug_imgs.append(
                    self.img_shift_crop(img, output_size, absulute_shift))
                bbox_shift = torch.tensor(
                    absulute_shift + [0, 0], device=init_bbox.device)
                aug_bboxes.append(init_bbox + bbox_shift)

        if 'fliplr' in augs and augs['fliplr']:
            shift = get_rand_shift()
            aug_imgs.append(
                self.img_shift_crop(img.flip(3), output_size, shift))
            bbox_shift = torch.tensor(shift + [0, 0], device=init_bbox.device)
            aug_bboxes.append(init_bbox + bbox_shift)

        if 'blur' in augs:
            for sigma in augs['blur']:
                kernel_size = [math.ceil(2 * s) for s in sigma]
                img_blur = gauss_blur(
                    img, kernel_size=kernel_size, sigma=sigma)
                shift = get_rand_shift()
                aug_imgs.append(
                    self.img_shift_crop(img_blur, output_size, shift))
                bbox_shift = torch.tensor(
                    shift + [0, 0], device=init_bbox.device)
                aug_bboxes.append(init_bbox + bbox_shift)

        if 'rotate' in augs:
            for angle in augs['rotate']:
                img_numpy = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
                assert img_numpy.ndim == 3
                rotated_img = imrotate(
                    img_numpy, angle, border_mode='replicate')
                img_tensor = torch.from_numpy(rotated_img.transpose(
                    2, 0, 1)).float().unsqueeze(0).to(img.device)
                shift = get_rand_shift()
                aug_imgs.append(
                    self.img_shift_crop(img_tensor, output_size, shift))
                bbox_shift = torch.tensor(
                    shift + [0, 0], device=init_bbox.device)
                aug_bboxes.append(init_bbox + bbox_shift)

        if 'dropout' in augs:
            for _ in range(len(augs['dropout'])):
                aug_bboxes.append(init_bbox)

        aug_imgs = torch.cat(aug_imgs, dim=0)
        aug_bboxes = torch.stack(aug_bboxes)
        return aug_imgs, aug_bboxes

    def generate_bbox(self, bbox: Tensor, sample_center: Tensor,
                      resize_factor: float) -> Tensor:
        """All inputs are based in original image coordinates and the outputs
        are based on the resized cropped image sample.

        Args:
            bbox (Tensor): of shape (4,) in [cx, cy, w, h] format
            sample_center (Tensor): of shape (2,)
            resize_factor (float):

        Return:
            Tensor: in [cx, cy, w, h] format
        """
        bbox_center = (bbox[:2] - sample_center) / resize_factor + (
            self.sample_size / 2)
        bbox_size = bbox[2:4] / resize_factor
        return torch.cat([bbox_center, bbox_size])

    def track(self, img: Tensor, data_samples: SampleList) -> InstanceList:
        """Track the box `bbox` of previous frame to current frame `img`.

        Args:
            img (Tensor): of shape (1, C, H, W).
            bbox (list | Tensor): The bbox in previous frame. The shape of the
                bbox is (4, ) in [cx, cy, w, h] format.

        Returns:
            conf_score (int): The confidence of predicted bbox.
            bbox (Tensor): The predicted bbox in [cx, cy, w, h] format
        """
        self.frame_num += 1
        bbox = self.memo.bbox.clone()

        # 1. Extract backbone features
        img_patch, patch_coord = self.get_cropped_img(
            img,
            bbox.round(),
            self.test_cfg['search_scale_factor'],
            self.sample_size,
            border_mode=self.test_cfg['border_mode'],
            max_scale_change=self.test_cfg['patch_max_scale_change'])

        backbone_feats = self.backbone(img_patch)

        # location of sample
        sample_center = patch_coord[:, :2].squeeze()
        sample_scale_factor = (patch_coord[:, 2:] /
                               self.sample_size).prod(dim=1).sqrt()

        # 2. Locate the target roughly using score map.
        new_bbox_center, score_map, state = self.classifier.predict(
            backbone_feats, data_samples, bbox, sample_center,
            sample_scale_factor)

        # 3. Refine position and scale of the target.
        if state != 'not_found':
            inside_offset = (self.test_cfg['bbox_inside_ratio'] -
                             0.5) * bbox[2:4]
            # clip the coordinates of the center of the target on the original
            # image
            bbox[:2] = torch.max(
                torch.min(new_bbox_center, self.img_size - inside_offset),
                inside_offset)

            cls_bboxes = self.generate_bbox(bbox, sample_center,
                                            sample_scale_factor)
            new_bbox = self.bbox_regressor.predict(backbone_feats,
                                                   data_samples, cls_bboxes,
                                                   sample_center,
                                                   sample_scale_factor)
            if new_bbox is not None:
                bbox = new_bbox

        # 4. Update the classifier
        update_flag = state not in ['not_found', 'uncertain']
        # Update the classifier filter using the latest position and size of
        # target
        if update_flag:
            # Create the target_bbox using the refined predicted boxes
            target_bbox = self.generate_bbox(bbox, sample_center,
                                             sample_scale_factor)
            hard_neg_flag = (state == 'hard_negative')
            # Update the filter of classifier using it's optimizer module
            self.classifier.update_classifier(target_bbox, self.frame_num,
                                              hard_neg_flag)

        result = [InstanceData()]
        result[0].scores = torch.max(score_map[0]).unsqueeze(0)
        result[0].bboxes = bbox_cxcywh_to_xyxy(bbox.unsqueeze(0))

        return result

    def get_cropped_img(
            self,
            img: Tensor,
            target_bbox: Tensor,
            search_scale_factor: float,
            output_size: Optional[Tensor] = None,
            border_mode: str = 'replicate',
            max_scale_change: Optional[float] = None) -> Tuple[Tensor, Tensor]:
        """Get the cropped patch based on the original image.

        Args:
            img (Tensor): The original image.
            target_bbox (Tensor): The bbox of target in [cx, cy, w,h] format.
            search_scale_factor (float): The ratio of cropped size to the size
                of target bbox.
            output_size (Optional[Tensor], optional): The output size.
                Defaults to None.
            border_mode (str, optional): The border mode. Defaults to
                'replicate'.
            max_scale_change (Optional[float], optional): The max scale change.
                Defaults to None.
            is_mask (Optional[bool], optional): Whether is mask.
                Defaults to False.

        Returns:
            img_patch (Tensor): of (1, c, h, w) shape.
            patch_coord (Tensor): of (1, 4) shape in [cx, cy, w, h] format.
        """
        crop_size = target_bbox[2:4].prod().sqrt() * search_scale_factor

        # Get new sample size if forced inside the image
        if border_mode == 'inside' or border_mode == 'inside_major':
            img_sz = torch.Tensor([img.shape[3],
                                   img.shape[2]]).to(target_bbox.device)
            shrink_factor = (crop_size.float() / img_sz)
            if border_mode == 'inside':
                shrink_factor = shrink_factor.max()
            elif border_mode == 'inside_major':
                shrink_factor = shrink_factor.min()
            shrink_factor.clamp_(min=1, max=max_scale_change)
            crop_size = (crop_size.float() / shrink_factor).long()

        tl = (target_bbox[:2] - crop_size // 2).long()
        br = (target_bbox[:2] + crop_size // 2).long()

        # Shift the crop to inside
        if border_mode == 'inside' or border_mode == 'inside_major':
            img2_sz = torch.LongTensor([img.shape[3],
                                        img.shape[2]]).to(target_bbox.device)
            shift = (-tl).clamp(0) - (br - img2_sz).clamp(0)
            tl += shift
            br += shift

            outside = torch.floor_divide(
                ((-tl).clamp(0) + (br - img2_sz).clamp(0)), 2)
            shift = (-tl - outside) * (outside > 0).long()
            tl += shift
            br += shift

        patch_coord = torch.cat((tl, br)).view(1, 4)
        patch_coord = bbox_xyxy_to_cxcywh(patch_coord)

        # Crop image patch
        img_patch = F.pad(
            img, (-tl[0].item(), br[0].item() - img.shape[3], -tl[1].item(),
                  br[1].item() - img.shape[2]),
            mode='replicate')

        if output_size is None:
            return img_patch.clone(), patch_coord

        # Resize
        img_patch = F.interpolate(
            img_patch,
            output_size.long().flip(0).tolist(),
            mode='bilinear',
            align_corners=True)

        return img_patch, patch_coord

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
        # has [T * N, C, H, W] shape and the first N images cover the entire
        # mini-batch.
        search_img = search_img.transpose(1, 0).contiguous().view(
            -1, *search_img.shape[2:])

        template_img = batch_inputs['img']
        assert template_img.dim(
        ) == 5, 'The img must be 5D Tensor (N, T, C, H, W).'
        # has [T * N, C, H, W] shape and the first N images cover the entire
        # mini-batch.
        template_img = template_img.transpose(1, 0).contiguous().view(
            -1, *template_img.shape[2:])

        z_feat = self.backbone(template_img)
        x_feat = self.backbone(search_img)

        losses = dict()
        loss_cls = self.classifier.loss(z_feat, x_feat, batch_data_samples,
                                        **kwargs)

        loss_bbox = self.bbox_regressor.loss(z_feat, x_feat,
                                             batch_data_samples, **kwargs)
        losses.update(loss_cls)
        losses.update(loss_bbox)
        return losses
