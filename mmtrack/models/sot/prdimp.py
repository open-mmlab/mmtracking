# Copyright (c) OpenMMLab. All rights reserved.
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from mmdet.models.builder import build_backbone, build_head
from torchvision.transforms.functional import normalize

from mmtrack.core.utils import TensorList, augmentation
from ..builder import MODELS
from .base import BaseSingleObjectTracker


@MODELS.register_module()
class Prdimp(BaseSingleObjectTracker):
    """The DiMP network.

    args:
        backbone:  Backbone feature extractor network. Must return a
            dict of feature maps
        classifier:  Target classification module.
        bb_regressor:  Bounding box regression module.
        classification_layer:  Name of the backbone feature layer to use for
            classification.
        bb_regressor_layer:  Names of the backbone layers to use for bounding
            box regression.
    """

    def __init__(self,
                 backbone=None,
                 cls_head=None,
                 reg_head=None,
                 init_cfg=None,
                 test_cfg=None):
        super().__init__(init_cfg)
        self.backbone = build_backbone(backbone)
        self.classifier = build_head(cls_head)
        self.bbox_regressor = build_head(reg_head)

        self.test_cfg = test_cfg

    def init(self, im, init_bbox):
        # Initialize some stuff
        self.frame_num = 1
        # Time initialization
        tic = time.time()

        # Get target position and size
        self.bbox_yx = torch.Tensor([
            init_bbox[1] + (init_bbox[3] - 1) / 2,
            init_bbox[0] + (init_bbox[2] - 1) / 2
        ])
        self.bbox_hw = torch.Tensor([init_bbox[3], init_bbox[2]])

        # Set sizes
        self.image_hw = torch.Tensor([im.shape[2], im.shape[3]])
        sz = self.test_cfg['image_sample_size']
        self.img_sample_sz = torch.Tensor(
            [sz, sz] if isinstance(sz, int) else sz)

        # Set search area
        search_area = torch.prod(self.bbox_hw *
                                 self.test_cfg['search_area_scale']).item()
        # target_scale is the ratio of the size of cropped image to the size
        # of resized image
        self.target_scale = math.sqrt(
            search_area) / self.img_sample_sz.prod().sqrt()

        # Target size in base scale
        # base_target_sz is the size of bbox in resized image
        self.base_target_sz = self.bbox_hw / self.target_scale

        # Setup scale factors
        self.scale_factors = torch.Tensor(self.test_cfg['scale_factor'])

        # Setup scale bounds
        self.min_scale_factor = torch.max(10 / self.base_target_sz)
        self.max_scale_factor = torch.min(self.image_hw / self.base_target_sz)

        # generate bboxes on the resized input image
        init_bboxes = self.generate_bbox(self.bbox_yx, self.bbox_hw,
                                         self.bbox_yx.round(),
                                         self.target_scale)

        # Extract features of augmented sample, is a tuple [layer2, layer3]
        im_patches = self.generate_aug_imgs(im)

        # Extract initial backbone features
        im_patches = normalize(
            im_patches / 255,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225))
        with torch.no_grad():
            init_backbone_feats = self.backbone(im_patches)

        init_aug_cls_bboxes = self.init_target_bboxes(init_bboxes).to(
            init_backbone_feats[-1].device)

        # initialize the filter in the classifier
        cls_feat = self.classifier.init_classifier(
            init_backbone_feats[-1], init_aug_cls_bboxes,
            self.test_cfg['init_aug_cfg']['augmentation'].get('dropout', None))

        # get feature size and kernel sizes used for calculating the
        # center of samples
        # TODO: simplify these code
        self.feature_sz = torch.Tensor(list(cls_feat.shape[-2:]))
        self.kernel_size = self.classifier.kernel_size

        # Init memory
        self.classifier.init_memory(TensorList([cls_feat]))

        # Initialize IoUNet
        self.bbox_regressor.init_iou_net(
            init_backbone_feats,
            init_bboxes.to(init_backbone_feats[-1].device))

        out = {'time': time.time() - tic}
        return out

    def generate_aug_imgs(self, im):
        """Perform data augmentation to generate initial training samples.

        return list[Tensor]
        """
        self.init_sample_scale = self.target_scale
        global_shift = torch.zeros(2)

        self.init_sample_pos = self.bbox_yx.round()

        # Compute augmentation size
        aug_expansion_factor = self.test_cfg['init_aug_cfg'].get(
            'augmentation_expansion_factor', None)
        aug_expansion_sz = self.img_sample_sz.clone()
        aug_output_sz = None
        if aug_expansion_factor is not None and aug_expansion_factor != 1:
            aug_expansion_sz = (self.img_sample_sz *
                                aug_expansion_factor).long()
            aug_expansion_sz += (aug_expansion_sz -
                                 self.img_sample_sz.long()) % 2
            aug_expansion_sz = aug_expansion_sz.float()
            aug_output_sz = self.img_sample_sz.long().tolist()

        # Random shift for each sample
        random_shift_factor = self.test_cfg['init_aug_cfg'].get(
            'random_shift_factor', 0)

        # torch.manual_seed(1)
        get_rand_shift = (
            (torch.rand(2) - 0.5) * self.img_sample_sz * random_shift_factor +
            global_shift).long().tolist() if random_shift_factor > 0 else None

        # Always put identity transformation first, since it is the
        # unaugmented sample that is always used
        self.transforms = [
            augmentation.Identity(aug_output_sz,
                                  global_shift.long().tolist())
        ]

        augs = self.test_cfg['init_aug_cfg']['augmentation']

        # Add all augmentations
        if 'shift' in augs:
            self.transforms.extend([
                augmentation.Translation(shift, aug_output_sz,
                                         global_shift.long().tolist())
                for shift in augs['shift']
            ])
        if 'relativeshift' in augs:
            for shift in augs['relativeshift']:
                absulute_shift = (torch.Tensor(shift) * self.img_sample_sz /
                                  2).long().tolist()
                self.transforms.append(
                    augmentation.Translation(absulute_shift, aug_output_sz,
                                             global_shift.long().tolist()))
        if 'fliplr' in augs and augs['fliplr']:
            self.transforms.append(
                augmentation.FlipHorizontal(aug_output_sz, get_rand_shift))
        if 'blur' in augs:
            self.transforms.extend([
                augmentation.Blur(sigma, aug_output_sz, get_rand_shift)
                for sigma in augs['blur']
            ])
        if 'scale' in augs:
            self.transforms.extend([
                augmentation.Scale(scale_factor, aug_output_sz, get_rand_shift)
                for scale_factor in augs['scale']
            ])
        if 'rotate' in augs:
            self.transforms.extend([
                augmentation.Rotate(angle, aug_output_sz, get_rand_shift)
                for angle in augs['rotate']
            ])

        # Extract augmented image patches
        im_patch, _ = self.sample_patch(
            im, self.init_sample_pos,
            self.init_sample_scale * aug_expansion_sz, aug_expansion_sz)
        im_patches = torch.cat(
            [T(im_patch, is_mask=False) for T in self.transforms])

        if 'dropout' in augs:
            num, _ = augs['dropout']
            self.transforms.extend(self.transforms[:1] * num)

        return im_patches

    def generate_bbox(self, pos, sz, sample_pos, sample_scale):
        """All inputs in original image coordinates.

        Generates a box in the cropped image sample reference frame, in the
            format used by the IoUNet.
        pos: bbox center
        sz: the width and height of the image
        sample_pos: the center of the sampled image
        """
        box_center = (pos - sample_pos) / sample_scale + (self.img_sample_sz -
                                                          1) / 2
        box_sz = sz / sample_scale
        target_ul = box_center - (box_sz - 1) / 2
        return torch.cat([target_ul.flip((0, )), box_sz.flip((0, ))])

    def init_target_bboxes(self, init_bboxes):
        """Get the target bounding boxes for the initial augmented samples."""
        init_target_boxes = TensorList()
        for T in self.transforms:
            init_target_boxes.append(
                init_bboxes + torch.Tensor([T.shift[1], T.shift[0], 0, 0]))
        init_target_boxes = torch.cat(init_target_boxes.view(1, 4), 0)

        return init_target_boxes

    def track(self, im):
        self.frame_num += 1

        # Extract backbone features
        centered_sample_pos = self.bbox_yx + (
            (self.feature_sz + self.kernel_size) % 2) * (
                self.target_scale * self.img_sample_sz) / (2 * self.feature_sz)

        # `im_patches` is of (num_scales, c, h, w) shape
        # `sample_coords` is of (num_scales, 4) shape.
        im_patches, sample_coords = self.sample_patch_multiscale(
            im,
            centered_sample_pos,
            self.target_scale * self.scale_factors,
            self.img_sample_sz,
            mode=self.test_cfg.get('border_mode', 'replicate'),
            max_scale_change=self.test_cfg.get('patch_max_scale_change', None))

        im_patches = normalize(
            im_patches / 255,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225))
        with torch.no_grad():
            backbone_feat = self.backbone(im_patches)

        # Extract classification features
        with torch.no_grad():
            test_x = self.classifier.cls_feature_extractor(backbone_feat[-1])

        # Location of sample
        sample_coords = sample_coords.float()
        sample_pos = 0.5 * (sample_coords[:, :2] + sample_coords[:, 2:] - 1)
        sample_scales = ((sample_coords[:, 2:] - sample_coords[:, :2]) /
                         self.img_sample_sz).prod(dim=1).sqrt()

        # Compute classification scores
        with torch.no_grad():
            scores_raw = self.classifier.classify(test_x)

        # Localize the target
        (translation_vec, scale_ind, scores,
         flag) = self.classifier.localize_target(scores_raw, sample_pos,
                                                 sample_scales,
                                                 self.img_sample_sz,
                                                 self.bbox_hw, self.bbox_yx)
        new_pos = sample_pos[scale_ind, :] + translation_vec

        # Refine position, size and  scale
        if flag != 'not_found':
            self.update_state(new_pos)
            cls_bboxes = self.generate_bbox(self.bbox_yx, self.bbox_hw,
                                            sample_pos[scale_ind, :],
                                            sample_scales[scale_ind])
            new_pos, new_target_sz = self.bbox_regressor.refine_target_box(
                cls_bboxes, backbone_feat, sample_pos[scale_ind, :],
                sample_scales[scale_ind], scale_ind)
            if new_pos is not None:
                new_scale = torch.sqrt(new_target_sz.prod() /
                                       self.base_target_sz.prod())
                self.bbox_yx_iounet = new_pos.clone()
                self.bbox_yx = new_pos.clone()
                self.bbox_hw = new_target_sz
                self.target_scale = new_scale

        # ------- UPDATE ------- #
        update_flag = flag not in ['not_found', 'uncertain']
        hard_negative = (flag == 'hard_negative')
        learning_rate = self.test_cfg.get('hard_negative_learning_rate',
                                          None) if hard_negative else None

        # update the classifier filter using the latest position and size of
        # target
        if update_flag:
            # Get train sample
            train_x = test_x[scale_ind:scale_ind + 1, ...]

            # Create target_box and label for spatial sample
            target_box = self.generate_bbox(self.bbox_yx, self.bbox_hw,
                                            sample_pos[scale_ind, :],
                                            sample_scales[scale_ind])

            # Update the classifier model
            self.classifier.update_classifier(train_x, target_box,
                                              self.frame_num, learning_rate,
                                              scores[scale_ind, ...])

        # Set the pos of the tracker to iounet pos
        if flag != 'not_found' and hasattr(self, 'bbox_yx_iounet'):
            self.bbox_yx = self.bbox_yx_iounet.clone()

        score_map = scores[scale_ind, ...]
        max_score = torch.max(score_map).item()

        # Compute output bounding box
        new_state = torch.cat(
            (self.bbox_yx[[1, 0]] - (self.bbox_hw[[1, 0]] - 1) / 2,
             self.bbox_hw[[1, 0]]))

        return max_score, new_state

    def update_state(self, new_pos, new_scale=None):
        # Update scale
        if new_scale is not None:
            self.target_scale = new_scale.clamp(self.min_scale_factor,
                                                self.max_scale_factor)
            self.bbox_hw = self.base_target_sz * self.target_scale

        # Update pos
        inside_ratio = self.test_cfg.get('target_inside_ratio', 0.2)
        inside_offset = (inside_ratio - 0.5) * self.bbox_hw
        self.bbox_yx = torch.max(
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
            bbox_pred[2:] -= bbox_pred[:2]  # in [x,y,w,h] format
            self.init(img, bbox_pred)
            best_score = -1.
        else:
            best_score, bbox_pred = self.track(img)

        bbox_pred[2:] += bbox_pred[:2]
        results = dict()
        results['track_bboxes'] = np.concatenate(
            (bbox_pred.cpu().numpy(), np.array([best_score])))
        return results

    def sample_patch_multiscale(self,
                                im,
                                pos,
                                scales,
                                image_sz,
                                mode: str = 'replicate',
                                max_scale_change=None):
        """Extract image patches at multiple scales.

        args:
            im: Image.
            pos: Center position for extraction.
            scales: Image scales to extract image patches from.
            image_sz: Size to resize the image samples to
            mode: how to treat image borders:
                'replicate' (default), 'inside' or 'inside_major'
            max_scale_change: maximum allowed scale change
                when using 'inside' and 'inside_major' mode
        """
        if isinstance(scales, (int, float)):
            scales = [scales]

        # Get image patches
        patch_iter, coord_iter = zip(*(self.sample_patch(
            im,
            pos,
            s * image_sz,
            image_sz,
            mode=mode,
            max_scale_change=max_scale_change) for s in scales))
        im_patches = torch.cat(list(patch_iter))
        patch_coords = torch.cat(list(coord_iter))

        return im_patches, patch_coords

    def sample_patch(self,
                     im: torch.Tensor,
                     crop_center: torch.Tensor,
                     crop_size: torch.Tensor,
                     output_size: torch.Tensor = None,
                     mode: str = 'replicate',
                     max_scale_change=None,
                     is_mask=False):
        """Sample an image patch.

        args:
            im: Image
            crop_center: center position of crop
            crop_size: size to crop
            output_size: size to resize to
            mode: how to treat image borders: 'replicate' (default), 'inside'
                or 'inside_major'
            max_scale_change: maximum allowed scale change when using 'inside'
                and 'inside_major' mode
        """

        # if mode not in ['replicate', 'inside']:
        #     raise ValueError('Unknown border mode \'{}\'.'.format(mode))

        # copy and convert
        crop_center_copy = crop_center.long().clone()

        pad_mode = mode

        # Get new sample size if forced inside the image
        if mode == 'inside' or mode == 'inside_major':
            pad_mode = 'replicate'
            im_sz = torch.Tensor([im.shape[2], im.shape[3]])
            shrink_factor = (crop_size.float() / im_sz)
            if mode == 'inside':
                shrink_factor = shrink_factor.max()
            elif mode == 'inside_major':
                shrink_factor = shrink_factor.min()
            shrink_factor.clamp_(min=1, max=max_scale_change)
            crop_size = (crop_size.float() / shrink_factor).long()

        # Compute pre-downsampling factor
        if output_size is not None:
            resize_factor = torch.min(crop_size.float() /
                                      output_size.float()).item()
            df = int(max(int(resize_factor - 0.1), 1))
        else:
            df = int(1)

        sz = crop_size.float() / df  # new size

        # Do downsampling
        if df > 1:
            os = crop_center_copy % df  # offset
            crop_center_copy = (crop_center_copy - os) // df  # new position
            im2 = im[..., os[0].item()::df, os[1].item()::df]  # downsample
        else:
            im2 = im

        # compute size to crop
        szl = torch.max(sz.round(), torch.Tensor([2])).long()

        # Extract top and bottom coordinates
        tl = crop_center_copy - (szl - 1) // 2
        br = crop_center_copy + szl // 2 + 1

        # Shift the crop to inside
        if mode == 'inside' or mode == 'inside_major':
            im2_sz = torch.LongTensor([im2.shape[2], im2.shape[3]])
            shift = (-tl).clamp(0) - (br - im2_sz).clamp(0)
            tl += shift
            br += shift

            outside = ((-tl).clamp(0) + (br - im2_sz).clamp(0)) // 2
            shift = (-tl - outside) * (outside > 0).long()
            tl += shift
            br += shift

            # Get image patch
            # im_patch =
            # im2[...,tl[0].item():br[0].item(),tl[1].item():br[1].item()]

        # Get image patch
        if not is_mask:
            im_patch = F.pad(im2, (-tl[1].item(), br[1].item() - im2.shape[3],
                                   -tl[0].item(), br[0].item() - im2.shape[2]),
                             pad_mode)
        else:
            im_patch = F.pad(im2, (-tl[1].item(), br[1].item() - im2.shape[3],
                                   -tl[0].item(), br[0].item() - im2.shape[2]))

        # Get image coordinates
        patch_coord = df * torch.cat((tl, br)).view(1, 4)

        if output_size is None or (im_patch.shape[-2] == output_size[0]
                                   and im_patch.shape[-1] == output_size[1]):
            return im_patch.clone(), patch_coord

        # Resample
        if not is_mask:
            im_patch = F.interpolate(
                im_patch, output_size.long().tolist(), mode='bilinear')
        else:
            im_patch = F.interpolate(
                im_patch, output_size.long().tolist(), mode='nearest')

        return im_patch, patch_coord

    def forward_train(self, imgs, img_metas, search_img, search_img_metas,
                      **kwargs):
        pass
