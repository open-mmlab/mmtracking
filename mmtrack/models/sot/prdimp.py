# Copyright (c) OpenMMLab. All rights reserved.
import math

import numpy as np
import torch
import torch.nn.functional as F
from mmdet.models.builder import build_backbone, build_head
from torchvision.transforms.functional import normalize

from mmtrack.core.utils import augmentation
from ..builder import MODELS
from .base import BaseSingleObjectTracker


@MODELS.register_module()
class Prdimp(BaseSingleObjectTracker):
    """PrDiMP: Probabilistic Regression for Visual Tracking.

    This single object tracker is the implementation of `PrDiMP
    <https://arxiv.org/abs/2003.12565>`_.

    args:
        backbone (dict): the configuration of backbone network.
        cls_head (dict, optional):  target classification module.
        bbox_head (dict, optional):  bounding box regression module.
        init_cfg (dict, optional): the configuration of initialization.
            Defaults to None.
        test_cfg (dict, optional): the configuration of test.
            Defaults to None.

    """

    def __init__(self,
                 backbone=None,
                 cls_head=None,
                 bbox_head=None,
                 init_cfg=None,
                 test_cfg=None):
        super().__init__(init_cfg)
        self.backbone = build_backbone(backbone)
        self.classifier = build_head(cls_head)
        self.bbox_regressor = build_head(bbox_head)
        self.test_cfg = test_cfg

    def init(self, img, init_bbox):
        self.frame_num = 1

        # initilatize some important global parameters in tracking
        self.init_params(img.shape[-2:], init_bbox)

        # generate bboxes on the resized input image
        # `init_bboxes` is in [x,y,w,h] format
        # TODO simplify this code
        init_bboxes = self.generate_bbox(self.bbox_center_yx, self.bbox_hw,
                                         self.bbox_center_yx.round(),
                                         self.target_scale)

        # Crop patches from the image and perform augmentation on the image
        # patches
        aug_img_patches = self.gen_aug_patches(img)

        # Extract initial backbone features
        aug_img_patches = normalize(
            aug_img_patches / 255,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225))

        # `init_backbone_feats` is a tuple containing the features of `layer2`
        # and `layer3`
        with torch.no_grad():
            init_backbone_feats = self.backbone(aug_img_patches)

        # Initialize the classifier with bboxes and features of `layer3`
        # get the augmented bboxes on the augmented image patches
        init_aug_cls_bboxes = self.init_cls_bboxes(init_bboxes).to(
            init_backbone_feats[-1].device)
        cls_feat_size = self.classifier.init_classifier(
            init_backbone_feats[-1], init_aug_cls_bboxes,
            self.test_cfg['init_aug_cfg']['augmentation']['dropout'])

        # Get feature size and kernel sizes used for calculating the
        # center of samples
        self.feature_sz = torch.Tensor(cls_feat_size)
        self.filter_size = self.classifier.filter_size

        # Initialize IoUNet
        # only use the features of the non-augmented image
        init_iou_features = [x[:1] for x in init_backbone_feats]
        self.bbox_regressor.init_iou_net(
            init_iou_features, init_bboxes.to(init_backbone_feats[-1].device))

    def init_params(self, img_hw, init_bbox):
        """Initilatize some important global parameters in tracking.

        Args:
            img_hw (ndarray): the height and width of image
            init_bbox (ndarray | list): in [x,y,w,h] format
        """
        # Set center and size for bbox
        self.bbox_center_yx = torch.Tensor([
            init_bbox[1] + (init_bbox[3] - 1) / 2,
            init_bbox[0] + (init_bbox[2] - 1) / 2
        ])
        self.bbox_hw = torch.Tensor([init_bbox[3], init_bbox[2]])

        # Set size for image
        self.image_hw = torch.Tensor([img_hw[0], img_hw[1]])
        sample_size = self.test_cfg['image_sample_size']
        self.img_sample_sz = torch.Tensor(
            [sample_size, sample_size] if isinstance(sample_size, int
                                                     ) else sample_size)

        # Set search area
        search_area = torch.prod(self.bbox_hw *
                                 self.test_cfg['search_area_scale']).item()
        # target_scale is the ratio of the size of cropped image to the size
        # of resized image
        self.target_scale = math.sqrt(
            search_area) / self.img_sample_sz.prod().sqrt()

        # base_target_sz is the size of the init bbox in resized image
        self.base_target_sz = self.bbox_hw / self.target_scale

        # Set scale bounds
        self.min_scale_factor = torch.max(10 / self.base_target_sz)
        self.max_scale_factor = torch.min(self.image_hw / self.base_target_sz)

    def gen_aug_patches(self, im):
        """Perform data augmentation to generate initial training samples and
        crop the patches from the augmented images.

        im ():
        """
        # Compute augmentation size
        aug_expansion_factor = self.test_cfg['init_aug_cfg'].get(
            'aug_expansion_factor', None)
        aug_expansion_sz = self.img_sample_sz.clone()
        aug_output_sz = None
        if aug_expansion_factor is not None and aug_expansion_factor != 1:
            aug_expansion_sz = (self.img_sample_sz *
                                aug_expansion_factor).long()
            # keep the same parity with `self.img_sample_sz`
            # TODO: verifiy the necessity of these code
            aug_expansion_sz += (aug_expansion_sz -
                                 self.img_sample_sz.long()) % 2
            aug_expansion_sz = aug_expansion_sz.float()
            aug_output_sz = self.img_sample_sz.long().tolist()

        random_shift_factor = self.test_cfg['init_aug_cfg'].get(
            'random_shift_factor', 0)

        def get_rand_shift():
            return ((torch.rand(2) - 0.5) * self.img_sample_sz *
                    random_shift_factor).long().tolist()

        # max_rand_shift = self.img_sample_sz.prod().sqrt() * \
        # random_shift_factor * 0.5

        # Always put identity transformation first, since it is the
        # unaugmented sample that is always used
        self.transforms = [augmentation.Identity(aug_output_sz)]

        augs = self.test_cfg['init_aug_cfg']['augmentation']

        # Add all augmentations
        if 'shift' in augs:
            self.transforms.extend([
                augmentation.Translation(shift, aug_output_sz)
                for shift in augs['shift']
            ])
        if 'relativeshift' in augs:
            for shift in augs['relativeshift']:
                absulute_shift = (torch.Tensor(shift) * self.img_sample_sz /
                                  2).long().tolist()
                self.transforms.append(
                    augmentation.Translation(absulute_shift, aug_output_sz))

        if 'fliplr' in augs and augs['fliplr']:
            self.transforms.append(
                augmentation.FlipHorizontal(aug_output_sz, get_rand_shift()))
        if 'blur' in augs:
            self.transforms.extend([
                augmentation.Blur(sigma, aug_output_sz, get_rand_shift())
                for sigma in augs['blur']
            ])
        if 'scale' in augs:
            self.transforms.extend([
                augmentation.Scale(scale_factor, aug_output_sz,
                                   get_rand_shift())
                for scale_factor in augs['scale']
            ])
        if 'rotate' in augs:
            self.transforms.extend([
                augmentation.Rotate(angle, aug_output_sz, get_rand_shift())
                for angle in augs['rotate']
            ])

        # Crop image patches
        img_patch, _ = self.get_cropped_img(
            im, self.bbox_center_yx.round(),
            self.target_scale * aug_expansion_sz, aug_expansion_sz)

        # Perform augmentation on the image patches
        aug_img_patches = torch.cat(
            [T(img_patch, is_mask=False) for T in self.transforms])

        if 'dropout' in augs:
            self.transforms.extend(self.transforms[:1] * len(augs['dropout']))

        return aug_img_patches

    def generate_bbox(self, pos, sz, sample_pos, sample_scale):
        """All inputs in original image coordinates.

        Args:
            Generates a box in the cropped image sample reference frame, in the
                format used by the IoUNet.
            pos: bbox center
            sz: the width and height of the image, with shape (2, )
            sample_pos (Tensor): of shape (2,) the center of the sampled image
            sample_scale (int):

        Return:
            bbox in [x,y,w,h] format
        """
        box_center = (pos - sample_pos) / sample_scale + (self.img_sample_sz -
                                                          1) / 2
        box_sz = sz / sample_scale
        target_ul = box_center - (box_sz - 1) / 2
        return torch.cat([target_ul.flip((0, )), box_sz.flip((0, ))])

    def init_cls_bboxes(self, init_bboxes):
        """Get the target bounding boxes for the initial augmented samples."""
        init_target_bboxes = []
        for T in self.transforms:
            init_target_bboxes.append(
                init_bboxes + torch.Tensor([T.shift[1], T.shift[0], 0, 0]))
        init_target_bboxes = torch.stack(init_target_bboxes)

        return init_target_bboxes

    def track(self, img):
        self.frame_num += 1

        # Extract backbone features
        centered_sample_pos = self.bbox_center_yx + (
            (self.feature_sz + self.filter_size) % 2) * (
                self.target_scale * self.img_sample_sz) / (2 * self.feature_sz)

        # `img_patch` is of (1, c, h, w) shape
        # `patch_coord` is of (1, 4) shape.
        img_patch, patch_coord = self.get_cropped_img(
            img,
            centered_sample_pos,
            self.target_scale * self.img_sample_sz,
            self.img_sample_sz,
            mode=self.test_cfg.get('border_mode', 'replicate'),
            max_scale_change=self.test_cfg.get('patch_max_scale_change', None))

        img_patch = normalize(
            img_patch / 255,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225))
        with torch.no_grad():
            backbone_feat = self.backbone(img_patch)

        # Extract classification features
        with torch.no_grad():
            test_x = self.classifier.cls_feature_extractor(backbone_feat[-1])
            scores_raw = self.classifier.classify(test_x)
            scores = torch.softmax(
                scores_raw.view(-1), dim=0).view(scores_raw.shape)

        # Location of sample
        patch_coord = patch_coord.float()
        sample_pos = 0.5 * (patch_coord[:, :2] + patch_coord[:, 2:] - 1)
        sample_scales = ((patch_coord[:, 2:] - patch_coord[:, :2]) /
                         self.img_sample_sz).prod(dim=1).sqrt()

        # Localize the target
        translation_vec, flag = self.classifier.localize_target(
            scores, sample_pos, sample_scales, self.img_sample_sz,
            self.bbox_hw, self.bbox_center_yx)
        new_pos = sample_pos[0, :] + translation_vec

        # Refine position, size and get new target scale
        if flag != 'not_found':
            self.update_state(new_pos)
            cls_bboxes = self.generate_bbox(self.bbox_center_yx, self.bbox_hw,
                                            sample_pos[0], sample_scales[0])
            new_pos, new_target_sz = self.bbox_regressor.refine_target_bbox(
                cls_bboxes, backbone_feat, sample_pos, sample_scales,
                self.img_sample_sz)
            if new_pos is not None:
                new_scale = torch.sqrt(new_target_sz.prod() /
                                       self.base_target_sz.prod())
                self.bbox_center_yx = new_pos.clone()
                self.bbox_hw = new_target_sz
                self.target_scale = new_scale

        # Update the classifier
        update_flag = flag not in ['not_found', 'uncertain']
        # Update the classifier filter using the latest position and size of
        # target
        if update_flag:
            # Create the target_bbox using the refined predicted boxes
            target_bbox = self.generate_bbox(self.bbox_center_yx, self.bbox_hw,
                                             sample_pos[0], sample_scales[0])
            hard_neg_flag = (flag == 'hard_negative')
            # Update the classifier model using it's optimizer module
            self.classifier.update_classifier(test_x, target_bbox,
                                              self.frame_num, hard_neg_flag)

        max_score = torch.max(scores[0]).item()
        # Compute output bounding box
        new_state = torch.cat(
            (self.bbox_center_yx[[1, 0]] - (self.bbox_hw[[1, 0]] - 1) / 2,
             self.bbox_hw[[1, 0]]))

        return max_score, new_state

    def update_state(self, new_pos, new_scale=None):
        """_summary_

        Args:
            new_pos (_type_): _description_
            new_scale (_type_, optional): _description_. Defaults to None.
        """
        # Update scale
        if new_scale is not None:
            self.target_scale = new_scale.clamp(self.min_scale_factor,
                                                self.max_scale_factor)
            self.bbox_hw = self.base_target_sz * self.target_scale

        # Update pos
        inside_ratio = self.test_cfg.get('target_inside_ratio', 0.2)
        inside_offset = (inside_ratio - 0.5) * self.bbox_hw
        self.bbox_center_yx = torch.max(
            torch.min(new_pos, self.image_hw - inside_offset), inside_offset)

    def simple_test(self, img, img_metas, gt_bboxes, **kwargs):
        """Test without augmentation.

        Args:
            img (Tensor): input image of shape (1, C, H, W).
            img_metas (list[dict]): list of image information dict where each
                dict has: 'img_shape', 'scale_factor', 'flip', and may also
                contain 'filename', 'ori_shape', 'pad_shape', and
                'img_norm_cfg'. For details on the values of these keys see
                `mmtrack/datasets/pipelines/formatting.py:VideoCollect`.
            gt_bboxes (list[Tensor]): list of ground truth bboxes for each
                image with shape (1, 4) in [tl_x, tl_y, br_x, br_y] format.

        Returns:
            dict(str : ndarray): the tracking results.
        """
        frame_id = img_metas[0].get('frame_id', -1)
        assert frame_id >= 0
        assert len(img) == 1, 'only support batch_size=1 when testing'
        self.frame_id = frame_id

        if frame_id == 0:
            bbox_pred = gt_bboxes[0][0]
            bbox_pred[2:] -= bbox_pred[:2]  # in [x, y, w, h] format
            self.init(img, bbox_pred)
            best_score = -1.
        else:
            best_score, bbox_pred = self.track(img)

        bbox_pred[2:] += bbox_pred[:2]  # in [x1, y1, x2, y2] format
        results = dict()
        results['track_bboxes'] = np.concatenate(
            (bbox_pred.cpu().numpy(), np.array([best_score])))
        return results

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
            2. Crop the image to a designated size (by the way of
                `torch.nn.functional.pad`)
            3. Reize the image to the `output_size`

        Args:
            img: Image
            crop_center: center position of crop
            crop_size: size to crop
            output_size: size to resize to
            mode: how to treat image borders: 'replicate' (default), 'inside'
                or 'inside_major'
            max_scale_change: maximgum allowed scale change when using 'inside'
                and 'inside_major' mode

        Returns:
            img_patch (Tensor): of shape (1, c, h, w)
            patch_coord (Tensor): the coordinate of image patch among the
                original image. It's of shape (1, 4) in [x1,y1,x2,y2] format.
        """
        # copy and convert
        crop_center_copy = crop_center.long().clone()

        pad_mode = mode

        # Get new sample size if forced inside the image
        if mode == 'inside' or mode == 'inside_major':
            pad_mode = 'replicate'
            img_sz = torch.Tensor([img.shape[2], img.shape[3]])
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
            df = int(max(int(resize_factor - 0.1), 1))
        else:
            df = int(1)
        sz = crop_size.float() / df  # new image size

        # Do downsampling
        if df > 1:
            os = crop_center_copy % df  # offset
            crop_center_copy = (crop_center_copy - os) // df  # new position
            img2 = img[..., os[0].item()::df, os[1].item()::df]  # downsample
        else:
            img2 = img

        # compute size to crop, size>=2
        szl = torch.max(sz.round(), torch.Tensor([2])).long()

        # Extract top and bottom coordinates
        # tl = crop_center_copy - torch.div(szl - 1, 2, rounding_mode='floor')
        # br = crop_center_copy + torch.div(szl, 2, rounding_mode='floor') + 1
        tl = crop_center_copy - (szl - 1) // 2
        br = crop_center_copy + szl // 2 + 1

        # Shift the crop to inside
        if mode == 'inside' or mode == 'inside_major':
            img2_sz = torch.LongTensor([img2.shape[2], img2.shape[3]])
            shift = (-tl).clamp(0) - (br - img2_sz).clamp(0)
            tl += shift
            br += shift

            outside = ((-tl).clamp(0) + (br - img2_sz).clamp(0)) // 2
            shift = (-tl - outside) * (outside > 0).long()
            tl += shift
            br += shift

        # Get image patch
        if not is_mask:
            img_patch = F.pad(img2,
                              (-tl[1].item(), br[1].item() - img2.shape[3],
                               -tl[0].item(), br[0].item() - img2.shape[2]),
                              pad_mode)
        else:
            img_patch = F.pad(img2,
                              (-tl[1].item(), br[1].item() - img2.shape[3],
                               -tl[0].item(), br[0].item() - img2.shape[2]))

        # Get image coordinates
        patch_coord = df * torch.cat((tl, br)).view(1, 4)

        if output_size is None or (img_patch.shape[-2] == output_size[0]
                                   and img_patch.shape[-1] == output_size[1]):
            return img_patch.clone(), patch_coord

        # Resample
        if not is_mask:
            img_patch = F.interpolate(
                img_patch, output_size.long().tolist(), mode='bilinear')
        else:
            img_patch = F.interpolate(
                img_patch, output_size.long().tolist(), mode='nearest')

        return img_patch, patch_coord

    def forward_train(self, imgs, img_metas, search_img, search_img_metas,
                      **kwargs):
        pass
