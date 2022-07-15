# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from mmdet.core.bbox.transforms import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from mmengine.data import InstanceData
from torch import Tensor
from torchvision.transforms.functional import gaussian_blur

from mmtrack.core.utils import (OptConfigType, OptMultiConfig, SampleList,
                                ndarray2tensor, rotate_image, tensor2ndarray)
from mmtrack.core.utils.typing import InstanceList
from mmtrack.registry import MODELS
from .base import BaseSingleObjectTracker


@MODELS.register_module()
class Prdimp(BaseSingleObjectTracker):
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
        super(Prdimp, self).__init__(data_preprocessor, init_cfg)
        self.backbone = MODELS.build(backbone)
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

        # initilatize some important global parameters in tracking
        self.init_params(img.shape[-2:], init_bbox)

        # Compute expanded size and output size about augmentation
        aug_expansion_factor = self.test_cfg['init_aug_cfg'][
            'aug_expansion_factor']
        aug_expansion_sz = self.sample_size.clone()
        aug_output_sz = None
        if aug_expansion_factor is not None and aug_expansion_factor != 1:
            aug_expansion_sz = (self.sample_size * aug_expansion_factor).long()
            # keep the same parity with `self.sample_size`
            # TODO: verifiy the necessity of these code
            # aug_expansion_sz += (aug_expansion_sz -
            #                      self.sample_size.long()) % 2
            aug_expansion_sz = aug_expansion_sz.float()
            aug_output_sz = self.sample_size.long().cpu().tolist()

        # Crop image patches and bboxes
        img_patch, _ = self.get_cropped_img(
            img, init_bbox[:2].round(), self.resize_factor * aug_expansion_sz,
            aug_expansion_sz)
        init_bbox = self.generate_bbox(init_bbox, init_bbox[:2].round(),
                                       self.resize_factor)

        # Crop patches from the image and perform augmentation on the image
        # patches
        aug_img_patches, aug_cls_bboxes = self.gen_aug_imgs_bboxes(
            img_patch, init_bbox, aug_output_sz)

        # Extract initial backbone features
        # aug_img_patches = normalize(
        #     aug_img_patches / 255,
        #     mean=(0.485, 0.456, 0.406),
        #     std=(0.229, 0.224, 0.225))

        # `init_backbone_feats` is a tuple containing the features of `layer2`
        # and `layer3`
        with torch.no_grad():
            init_backbone_feats = self.backbone(aug_img_patches)

        # Initialize the classifier with bboxes and features of `layer3`
        # get the augmented bboxes on the augmented image patches
        cls_feat_size = self.classifier.init_classifier(
            init_backbone_feats[-1], aug_cls_bboxes,
            self.test_cfg['init_aug_cfg']['augmentation']['dropout'])

        # Get feature size and kernel sizes used for calculating the
        # center of samples
        # TODO: remove these two variables
        self.cls_feat_size = torch.Tensor(cls_feat_size).to(init_bbox.device)
        self.filter_size = torch.Tensor(self.classifier.filter_size).to(
            init_bbox.device)

        # Initialize IoUNet
        # only use the features of the non-augmented image
        init_iou_features = [x[:1] for x in init_backbone_feats]
        self.bbox_regressor.init_iou_net(init_iou_features, init_bbox)

    def init_params(self, img_hw, init_bbox):
        """Initilatize some important global parameters in tracking.

        Args:
            img_hw (ndarray): the height and width of the image.
            init_bbox (ndarray | list): in [cx, cy, w, h] format.
        """
        # Set size for image. img_size is in [w, h] format.
        self.img_size = torch.Tensor([img_hw[1],
                                      img_hw[0]]).to(init_bbox.device)
        sample_size = self.test_cfg['image_sample_size']
        sample_size = torch.Tensor([sample_size, sample_size] if isinstance(
            sample_size, int) else sample_size)
        self.sample_size = sample_size.to(init_bbox.device)

        # Set search area
        search_area = torch.prod(init_bbox[2:4] *
                                 self.test_cfg['search_area_scale'])
        # resize_factor is the ratio of the size of original image to that
        # of the resized image. Correspondingly, the original bbox is also
        # resized by the same factor.
        self.resize_factor = search_area.sqrt() / self.sample_size.prod().sqrt(
        )

        # base_target_sz is the size of the init bbox in the resized image
        self.base_target_sz = init_bbox[2:4] / self.resize_factor

        # Set scale bounds
        self.min_scale_factor = torch.max(10 / self.base_target_sz)
        self.max_scale_factor = torch.min(self.img_size / self.base_target_sz)

    def img_shift_crop(self, img, output_size=None, shift=None):
        """Shift and crop the image.

        Args:
            img (Tensor): The image of shape (C, H, W).
            output_size (list): in [w, h] format.
            shift (list): in [x, y] fotmat.

        Returns:
            Tensor: Output image.
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

    def gen_aug_imgs_bboxes(self, img, init_bbox, output_size):
        """Perform data augmentation.

        Args:
            img (Tensor): Cropped image of shape (1, C, H, W).
            init_bbox (Tensor): of (4, ) shape in [cx, cy, w, h] format.
            output_size (Tensor): of (2, ) shape in [w, h] format.

        Returns:
            Tensor: The cropped augmented image patches.
        """
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
                kernel_size = [2 * s + 1 for s in sigma]
                img_blur = gaussian_blur(
                    img, kernel_size=kernel_size, sigma=sigma)
                shift = get_rand_shift()
                aug_imgs.append(
                    self.img_shift_crop(img_blur, output_size, shift))
                bbox_shift = torch.tensor(
                    shift + [0, 0], device=init_bbox.device)
                aug_bboxes.append(init_bbox + bbox_shift)

        if 'rotate' in augs:
            for angle in augs['rotate']:
                img_numpy = tensor2ndarray(img)
                assert img_numpy.ndim == 3
                rotated_img = rotate_image(
                    img_numpy,
                    angle,
                    border_mode='replicate',
                )
                img_tensor = ndarray2tensor(rotated_img, device=img.device)
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

    def generate_bbox(self, bbox, sample_center, sample_scale):
        """All inputs are based in original image coordinates and the outputs
        are based on the resized cropped image sample.

        Args:
            bbox (Tensor): of shape (4,) in [cx, cy, w, h] format
            sample_center (Tensor): of shape (2,)
            sample_scale (int):

        Return:
            Tensor: in [cx, cy, w, h] format
        """
        bbox_center = (bbox[:2] - sample_center) / sample_scale + (
            self.sample_size / 2)
        bbox_size = bbox[2:4] / sample_scale
        return torch.cat([bbox_center, bbox_size])

    def track(self, img: Tensor,
              batch_data_samples: SampleList) -> InstanceList:
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

        # Extract backbone features
        # TODO: remove `self.cls_feat_size`
        sample_center_crop = bbox[:2] + (
            (self.cls_feat_size + self.filter_size) % 2
        ) * (self.resize_factor * self.sample_size) / (2 * self.cls_feat_size)

        # `img_patch` is of (1, c, h, w) shape
        # `patch_coord` is of (1, 4) shape in [cx, cy, w, h] format.
        img_patch, patch_coord = self.get_cropped_img(
            img,
            sample_center_crop,
            self.resize_factor * self.sample_size,
            self.sample_size,
            mode=self.test_cfg['border_mode'],
            max_scale_change=self.test_cfg['patch_max_scale_change'])

        with torch.no_grad():
            backbone_feats = self.backbone(img_patch)

        # Location of sample
        sample_center = patch_coord[:, :2]
        sample_scales = (patch_coord[:, 2:] /
                         self.sample_size).prod(dim=1).sqrt()

        new_bbox_center, scores, test_feat, flag = self.classifier.predict(
            backbone_feats, bbox, patch_coord, self.sample_size)

        # Refine position, size and get new target scale
        if flag != 'not_found':
            # TODO: hard code here
            inside_ratio = 0.2
            inside_offset = (inside_ratio - 0.5) * bbox[2:4]
            # Clip the coordinates of the center of the target
            bbox[:2] = torch.max(
                torch.min(new_bbox_center, self.img_size - inside_offset),
                inside_offset)

            # TODO: unify the bbox format of classifier and the bbox regressor
            # and not re-generate bbox
            cls_bboxes = self.generate_bbox(bbox, sample_center[0],
                                            sample_scales[0])
            new_bbox = self.bbox_regressor.predict(cls_bboxes, backbone_feats,
                                                   sample_center,
                                                   sample_scales,
                                                   self.sample_size)
            if new_bbox is not None:
                # Crop the original image based on the `self.resize_factor`
                self.resize_factor = torch.sqrt(new_bbox[2:4].prod() /
                                                self.base_target_sz.prod())
                bbox = new_bbox

        # Update the classifier
        update_flag = flag not in ['not_found', 'uncertain']
        # Update the classifier filter using the latest position and size of
        # target
        if update_flag:
            # Create the target_bbox using the refined predicted boxes
            target_bbox = self.generate_bbox(bbox, sample_center[0],
                                             sample_scales[0])
            hard_neg_flag = (flag == 'hard_negative')
            # Update the filter of classifier using it's optimizer module
            self.classifier.update_classifier(test_feat, target_bbox,
                                              self.frame_num, hard_neg_flag)

        conf_score = torch.max(scores[0])
        # Compute output bounding box

        result = InstanceData()
        result.scores = conf_score.unsqueeze(0)
        result.bboxes = bbox_cxcywh_to_xyxy(bbox.unsqueeze(0))

        return result

    '''
    def get_cropped_img(self,
                        img: torch.Tensor,
                        crop_center: torch.Tensor,
                        crop_size: torch.Tensor,
                        output_size: torch.Tensor = None,
                        mode: str = 'replicate',
                        max_scale_change=None,
                        is_mask=False):
        """Crop an image patch from the original image. The target is centered
            at the cropped image patch.
        There are 3 steps:
            1. Downsample the image according to the ratio of `crop_size` to
                `output_size`
            2. Crop the image to a designated size (in the way of
                `torch.nn.functional.pad`)
            3. Reize the image to the `output_size`

        When mode is 'inside' or 'inside_major', the cropped patch may not be
        centered at the `crop_center`

        Args:
            img (Tensor): Input image.
            crop_center (Tensor): center position of crop in [x, y] format.
            crop_size (Tensor): size to crop in [w, h] format.
            output_size (Tensor): size to resize to in [w, h] format.
            mode: how to treat image borders: 'replicate' (default), 'inside'
                or 'inside_major'
            max_scale_change: maximgum allowed scale change when using 'inside'
                and 'inside_major' mode

        Returns:
            img_patch (Tensor): of shape (1, c, h, w)
            patch_coord (Tensor): the coordinate of image patch among the
                original image. It's of shape (1, 4) in [cx, cy, w, h] format.
        """

        # TODO: Simplify this preprocess
        # copy and convert
        crop_center_copy = crop_center.long().clone()
        pad_mode = mode
        # Get new sample size if forced inside the image
        if mode == 'inside' or mode == 'inside_major':
            pad_mode = 'replicate'
            img_sz = torch.Tensor([img.shape[3],
                                   img.shape[2]]).to(crop_size.device)
            shrink_factor = (crop_size.float() / img_sz)
            if mode == 'inside':
                shrink_factor = shrink_factor.max()
            elif mode == 'inside_major':
                shrink_factor = shrink_factor.min()
            shrink_factor.clamp_(min=1, max=max_scale_change)
            crop_size = (crop_size.float() / shrink_factor).long()

        # Compute pre-downsampling factor
        # requires resize_factor >= 1 and is type of int
        if output_size is not None:
            resize_factor = torch.min(crop_size.float() /
                                      output_size.float()).item()
            resize_factor = int(max(int(resize_factor - 0.1), 1))
        else:
            resize_factor = torch.Tensor(1)

        # Do downsampling to get `img2`
        if resize_factor > 1:
            offset = crop_center_copy % resize_factor  # offset
            crop_center_copy = torch.floor_divide(
                crop_center_copy - offset, resize_factor)  # new position
            img2 = img[..., offset[1].item()::resize_factor,
                       offset[0].item()::resize_factor]  # downsample
        else:
            img2 = img

        # cropped image size
        cropped_img_size = crop_size.float() / resize_factor
        cropped_img_size = torch.clamp_min(cropped_img_size.round(), 2).long()
        # Extract top and bottom coordinates
        tl = crop_center_copy - torch.floor_divide(cropped_img_size, 2)
        br = crop_center_copy + torch.floor_divide(cropped_img_size, 2)

        # Shift the crop to inside
        if mode == 'inside' or mode == 'inside_major':
            img2_sz = torch.LongTensor([img2.shape[3],
                                        img2.shape[2]]).to(crop_center.device)
            shift = (-tl).clamp(0) - (br - img2_sz).clamp(0)
            tl += shift
            br += shift

            outside = torch.floor_divide(
                ((-tl).clamp(0) + (br - img2_sz).clamp(0)), 2)
            shift = (-tl - outside) * (outside > 0).long()
            tl += shift
            br += shift

        # Crop image patch
        if not is_mask:
            img_patch = F.pad(img2,
                              (-tl[0].item(), br[0].item() - img2.shape[3],
                               -tl[1].item(), br[1].item() - img2.shape[2]),
                              pad_mode)
        else:
            img_patch = F.pad(img2,
                              (-tl[0].item(), br[0].item() - img2.shape[3],
                               -tl[1].item(), br[1].item() - img2.shape[2]))

        # Get image coordinates
        patch_coord = resize_factor * torch.cat((tl, br)).view(1, 4)
        patch_coord = bbox_xyxy_to_cxcywh(patch_coord)

        if output_size is None or (img_patch.shape[-2] == output_size[0]
                                   and img_patch.shape[-1] == output_size[1]):
            return img_patch.clone(), patch_coord

        # Resize
        if not is_mask:
            img_patch = F.interpolate(
                img_patch,
                output_size.long().flip(0).tolist(),
                mode='bilinear',
                align_corners=True)
        else:
            img_patch = F.interpolate(
                img_patch, output_size.long().flip(0).tolist(), mode='nearest')

        return img_patch, patch_coord
    '''

    def get_cropped_img(self,
                        img: torch.Tensor,
                        crop_center: torch.Tensor,
                        crop_size: torch.Tensor,
                        output_size: torch.Tensor = None,
                        mode: str = 'replicate',
                        max_scale_change=None,
                        is_mask=False):
        pad_mode = mode
        # Get new sample size if forced inside the image
        if mode == 'inside' or mode == 'inside_major':
            pad_mode = 'replicate'
            img_sz = torch.Tensor([img.shape[3],
                                   img.shape[2]]).to(crop_size.device)
            shrink_factor = (crop_size.float() / img_sz)
            if mode == 'inside':
                shrink_factor = shrink_factor.max()
            elif mode == 'inside_major':
                shrink_factor = shrink_factor.min()
            shrink_factor.clamp_(min=1, max=max_scale_change)
            crop_size = (crop_size.float() / shrink_factor).long()

        tl = (crop_center - crop_size // 2).long()
        br = (crop_center + crop_size // 2).long()

        # Shift the crop to inside
        if mode == 'inside' or mode == 'inside_major':
            img2_sz = torch.LongTensor([img.shape[3],
                                        img.shape[2]]).to(crop_center.device)
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
        if not is_mask:
            img_patch = F.pad(img,
                              (-tl[0].item(), br[0].item() - img.shape[3],
                               -tl[1].item(), br[1].item() - img.shape[2]),
                              pad_mode)
        else:
            img_patch = F.pad(img,
                              (-tl[0].item(), br[0].item() - img.shape[3],
                               -tl[1].item(), br[1].item() - img.shape[2]))

        if output_size is None or (img_patch.shape[-2] == output_size[0]
                                   and img_patch.shape[-1] == output_size[1]):
            return img_patch.clone(), patch_coord

        # Resize
        if not is_mask:
            img_patch = F.interpolate(
                img_patch,
                output_size.long().flip(0).tolist(),
                mode='bilinear',
                align_corners=True)
        else:
            img_patch = F.interpolate(
                img_patch, output_size.long().flip(0).tolist(), mode='nearest')

        return img_patch, patch_coord

    def loss(self, imgs, img_metas, search_img, search_img_metas, **kwargs):
        pass
