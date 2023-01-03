# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from mmcv.transforms import BaseTransform
from mmcv.transforms.utils import cache_randomness
from mmdet.datasets.transforms import RandomCrop as MMDET_RandomCrop
from mmengine.logging import print_log

from mmtrack.registry import TRANSFORMS
from mmtrack.utils import crop_image


@TRANSFORMS.register_module()
class CropLikeSiamFC(BaseTransform):
    """Crop images as SiamFC did.

    The way of cropping an image is proposed in
    "Fully-Convolutional Siamese Networks for Object Tracking."
    `SiamFC <https://arxiv.org/abs/1606.09549>`_.

    Required Keys:

    - gt_bboxes (np.float32)
    - gt_bboxes_labels (np.int32)
    - gt_instances_id (np.int32)
    - gt_masks (BitmapMasks | PolygonMasks)
    - gt_seg_map (np.uint8)
    - gt_ignore_flags (np.bool)
    - img
    - img_shape (optional)

    Modified Keys:

    - gt_bboxes
    - img
    - img_shape (optional)

    Args:
        context_amount (float): The context amount around a bounding box.
            Defaults to 0.5.
        exemplar_size (int): Exemplar size. Defaults to 127.
        crop_size (int): Crop size. Defaults to 511.
    """

    def __init__(self,
                 context_amount: float = 0.5,
                 exemplar_size: int = 127,
                 crop_size: int = 511):
        self.context_amount = context_amount
        self.exemplar_size = exemplar_size
        self.crop_size = crop_size

    def crop_like_SiamFC(self,
                         image: np.ndarray,
                         bbox: np.ndarray,
                         context_amount: float = 0.5,
                         exemplar_size: int = 127,
                         crop_size: int = 511) -> np.ndarray:
        """Crop an image as SiamFC did.

        Args:
            image (np.ndarray): of shape (H, W, 3).
            bbox (np.ndarray): of shape (4, ) in [x1, y1, x2, y2] format.
            context_amount (float): The context amount around a bounding box.
                Defaults to 0.5.
            exemplar_size (int): Exemplar size. Defaults to 127.
            crop_size (int): Crop size. Defaults to 511.

        Returns:
            np.ndarray: The cropped image of shape (crop_size, crop_size, 3).
        """
        padding = np.mean(image, axis=(0, 1)).tolist()

        bbox = np.array([
            0.5 * (bbox[2] + bbox[0]), 0.5 * (bbox[3] + bbox[1]),
            bbox[2] - bbox[0], bbox[3] - bbox[1]
        ])
        z_width = bbox[2] + context_amount * (bbox[2] + bbox[3])
        z_height = bbox[3] + context_amount * (bbox[2] + bbox[3])
        z_size = np.sqrt(z_width * z_height)

        z_scale = exemplar_size / z_size
        d_search = (crop_size - exemplar_size) / 2.
        pad = d_search / z_scale
        x_size = z_size + 2 * pad
        x_bbox = np.array([
            bbox[0] - 0.5 * x_size, bbox[1] - 0.5 * x_size,
            bbox[0] + 0.5 * x_size, bbox[1] + 0.5 * x_size
        ])

        x_crop_img = crop_image(image, x_bbox, crop_size, padding)
        return x_crop_img

    def generate_box(self, image: np.ndarray, gt_bbox: np.ndarray,
                     context_amount: float, exemplar_size: int) -> np.ndarray:
        """Generate box based on cropped image.

        Args:
            image (np.ndarray): The cropped image of shape
                (self.crop_size, self.crop_size, 3).
            gt_bbox (np.ndarray): In shape (4, ), in [x1, y1, x2, y2] format.
            context_amount (float): The context amount around a bounding box.
            exemplar_size (int): Exemplar size. Defaults to 127.

        Returns:
            np.ndarray: Generated box of shape (4, ) in [x1, y1, x2, y2]
                format.
        """
        img_h, img_w = image.shape[:2]
        w, h = gt_bbox[2] - gt_bbox[0], gt_bbox[3] - gt_bbox[1]

        z_width = w + context_amount * (w + h)
        z_height = h + context_amount * (w + h)
        z_scale = np.sqrt(z_width * z_height)
        z_scale_factor = exemplar_size / z_scale
        w = w * z_scale_factor
        h = h * z_scale_factor
        cx, cy = img_w // 2, img_h // 2
        bbox = np.array(
            [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h],
            dtype=np.float32)

        return bbox

    def transform(self, results: dict) -> dict:
        """The transform function.

        For each dict in results, crop image like SiamFC did.

        Args:
            results (dict): Dict from :obj:`mmtrack.dataset.BaseSOTDataset`.

        Returns:
            dict: Dict that contains cropped images and
                corresponding ground truth boxes.
        """
        crop_img = self.crop_like_SiamFC(results['img'],
                                         results['gt_bboxes'].squeeze(),
                                         self.context_amount,
                                         self.exemplar_size, self.crop_size)
        generated_bbox = self.generate_box(crop_img,
                                           results['gt_bboxes'].squeeze(),
                                           self.context_amount,
                                           self.exemplar_size)
        if 'img_shape' in results:
            results['img_shape'] = crop_img.shape

        results['img'] = crop_img
        results['gt_bboxes'] = generated_bbox[None]

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(context_amount={self.context_amount}, '
        repr_str += f'exemplar_size={self.exemplar_size}, '
        repr_str += f'crop_size={self.crop_size})'
        return repr_str


@TRANSFORMS.register_module()
class SeqCropLikeStark(BaseTransform):
    """Crop images as Stark did.

    The way of cropping an image is proposed in
    "Learning Spatio-Temporal Transformer for Visual Tracking."
    `Stark <https://arxiv.org/abs/2103.17154>`_.

    Required Keys:

    - gt_bboxes (np.float32)
    - gt_bboxes_labels (np.int32)
    - gt_instances_id (np.int32)
    - gt_masks (BitmapMasks | PolygonMasks)
    - gt_seg_map (np.uint8)
    - gt_ignore_flags (np.bool)
    - img
    - img_shape (optional)
    - jittered_bboxes (np.float32)

    Modified Keys:

    - gt_bboxes
    - img
    - img_shape (optional)

    Added keys:

    - padding_mask

    Args:
        crop_size_factor (list[int | float]): contains the ratio of crop size
            to bbox size.
        output_size (list[int | float]): contains the size of resized image
            (always square).
    """

    def __init__(self, crop_size_factor: List[Union[int, float]],
                 output_size: List[Union[int, float]]):
        self.crop_size_factor = crop_size_factor
        self.output_size = output_size

    def crop_like_stark(
            self, img: np.ndarray, bbox: np.ndarray,
            crop_size_factor: np.ndarray,
            output_size: int) -> Union[np.ndarray, float, np.ndarray]:
        """Crop an image as Stark did.

        Args:
            img (np.ndarray): of shape (H, W, 3).
            bbox (np.ndarray): of shape (4, ) in [x1, y1, x2, y2] format.
            crop_size_factor (float): the ratio of crop size to bbox size
            output_size (int): the size of resized image (always square).

        Returns:
            img_crop_padded (np.ndarray): the cropped image of shape
                (crop_size, crop_size, 3).
            resize_factor (float): the ratio of original image scale to cropped
                image scale.
            pdding_mask (np.ndarray): the padding mask caused by cropping.
        """
        x1, y1, x2, y2 = np.split(bbox, 4, axis=-1)
        bbox_w, bbox_h = x2 - x1, y2 - y1
        cx, cy = x1 + bbox_w / 2., y1 + bbox_h / 2.

        img_h, img_w, _ = img.shape
        # 1. Crop image
        # 1.1 calculate crop size and pad size
        crop_size = math.ceil(math.sqrt(bbox_w * bbox_h) * crop_size_factor)
        crop_size = max(crop_size, 1)

        x1 = int(np.round(cx - crop_size * 0.5))
        x2 = x1 + crop_size
        y1 = int(np.round(cy - crop_size * 0.5))
        y2 = y1 + crop_size

        x1_pad = max(0, -x1)
        x2_pad = max(x2 - img_w + 1, 0)
        y1_pad = max(0, -y1)
        y2_pad = max(y2 - img_h + 1, 0)

        # 1.2 crop image
        img_crop = img[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]

        # 1.3 pad image
        img_crop_padded = cv2.copyMakeBorder(img_crop, y1_pad, y2_pad, x1_pad,
                                             x2_pad, cv2.BORDER_CONSTANT)
        # 1.4 generate padding mask
        img_h, img_w, _ = img_crop_padded.shape
        pdding_mask = np.ones((img_h, img_w))
        end_x, end_y = -x2_pad, -y2_pad
        if y2_pad == 0:
            end_y = None
        if x2_pad == 0:
            end_x = None
        pdding_mask[y1_pad:end_y, x1_pad:end_x] = 0

        # 2. Resize image and padding mask
        resize_factor = output_size / crop_size
        img_crop_padded = cv2.resize(img_crop_padded,
                                     (output_size, output_size))
        pdding_mask = cv2.resize(pdding_mask,
                                 (output_size, output_size)).astype(np.bool_)

        return img_crop_padded, resize_factor, pdding_mask

    def generate_box(self,
                     bbox_gt: np.ndarray,
                     bbox_cropped: np.ndarray,
                     resize_factor: float,
                     output_size: float,
                     normalize: bool = False) -> np.ndarray:
        """Transform the box coordinates from the original image coordinates to
        the coordinates of the cropped image.

        Args:
            bbox_gt (np.ndarray): of shape (4, ) in [x1, y1, x2, y2] format.
            bbox_cropped (np.ndarray): of shape (4, ) in [x1, y1, x2, y2]
                format.
            resize_factor (float): the ratio of original image scale to cropped
                image scale.
            output_size (float): the size of output image.
            normalize (bool): whether to normalize the output box.
                Defaults to False.

        Returns:
            np.ndarray: generated box of shape (4, ) in [x1, y1, x2, y2]
                format.
        """
        assert output_size > 0
        bbox_gt_center = (bbox_gt[0:2] + bbox_gt[2:4]) * 0.5
        bbox_cropped_center = (bbox_cropped[0:2] + bbox_cropped[2:4]) * 0.5

        bbox_out_center = (output_size - 1) / 2. + (
            bbox_gt_center - bbox_cropped_center) * resize_factor
        bbox_out_wh = (bbox_gt[2:4] - bbox_gt[0:2]) * resize_factor
        bbox_out = np.concatenate((bbox_out_center - 0.5 * bbox_out_wh,
                                   bbox_out_center + 0.5 * bbox_out_wh),
                                  axis=-1)

        return bbox_out / output_size if normalize else bbox_out

    def transform(self, results: dict) -> dict:
        """The transform function. For each dict in results, crop image like
        Stark did.

        Args:
            results (dict): Dict of list from
                :obj:`mmtrack.datasets.SeqBboxJitter`.

        Returns:
            dict: Dict of list that contains cropped image and
                the corresponding groundtruth bbox.
        """
        imgs = results['img']
        gt_bboxes = results['gt_bboxes']
        jittered_bboxes = results['jittered_bboxes']
        new_imgs = []
        results['padding_mask'] = []
        for i, (img, gt_bbox, jittered_bbox) in enumerate(
                zip(imgs, gt_bboxes, jittered_bboxes)):
            gt_bbox, jittered_bbox = gt_bbox[0], jittered_bbox[0]
            crop_img, resize_factor, padding_mask = self.crop_like_stark(
                img, jittered_bbox, self.crop_size_factor[i],
                self.output_size[i])

            generated_bbox = self.generate_box(
                gt_bbox,
                jittered_bbox,
                resize_factor,
                self.output_size[i],
                normalize=False)

            new_imgs.append(crop_img)
            if 'img_shape' in results:
                results['img_shape'][i] = crop_img.shape
            results['gt_bboxes'][i] = generated_bbox[None]
            results['padding_mask'].append(padding_mask)

        results['img'] = new_imgs

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'crop_size_factor={self.crop_size_factor}, '
        repr_str += f'output_size={self.output_size})'
        return repr_str


@TRANSFORMS.register_module()
class CropLikeDiMP(BaseTransform):
    """Crop images as PrDiMP did.

    The way of cropping an image is proposed in
    "Learning Discriminative Model Prediction for Tracking."
    `DiMP <https://arxiv.org/abs/1904.07220>`_.

    Args:
        crop_size_factor (float): contains the ratio of crop size
            to bbox size.
        output_size (float): contains the size of resized image
            (always square).
    """

    def __init__(self, crop_size_factor: float, output_size: float):
        self.crop_size_factor = crop_size_factor
        self.output_size = output_size

    def crop_like_dimp(
            self, img: np.ndarray, bbox: np.ndarray, crop_size_factor: float,
            output_size: int) -> Tuple[np.ndarray, np.ndarray, float]:
        """Crop an image as DiMP did.

        Note: The difference between dimp and stark is the operation of moving
        box inside image in dimp. This may cause the cropped image is not
        centered on the `bbox`.

        Args:
            image (np.ndarray): of shape (H, W, 3).
            bbox (np.ndarray): of shape (4, ) in [x1, y1, x2, y2] format.
            crop_size_factor (float): the ratio of crop size to bbox size
            output_size (int): the size of resized image (always square).

        Returns:
            img_crop_padded (np.ndarray): the cropped image of shape
                (crop_size, crop_size, 3).
            resize_factor (float): the ratio of original image scale to cropped
                image scale.
            pdding_mask (np.ndarray): the padding mask caused by cropping.
        """
        x1, y1, x2, y2 = np.split(bbox, 4, axis=-1)
        bbox_w, bbox_h = x2 - x1, y2 - y1
        cx, cy = x1 + bbox_w / 2., y1 + bbox_h / 2.

        img_h, img_w, _ = img.shape
        # 1. Crop image
        # 1.1 calculate crop size
        crop_size = math.ceil(math.sqrt(bbox_w * bbox_h) * crop_size_factor)
        crop_size = max(crop_size, 1)

        x1 = int(np.round(cx - crop_size * 0.5))
        x2 = x1 + crop_size
        y1 = int(np.round(cy - crop_size * 0.5))
        y2 = y1 + crop_size

        # 1.2 Move box inside image
        shift_x = max(0, -x1) + min(0, img_w - x2)
        x1 += shift_x
        x2 += shift_x

        shift_y = max(0, -y1) + min(0, img_h - y2)
        y1 += shift_y
        y2 += shift_y

        # keep the balance of left and right spacing if crop area exceeds the
        # image
        out_x = (max(0, -x1) + max(0, x2 - img_w)) // 2
        out_y = (max(0, -y1) + max(0, y2 - img_h)) // 2
        shift_x = (-x1 - out_x) * (out_x > 0)
        shift_y = (-y1 - out_y) * (out_y > 0)

        x1 += shift_x
        x2 += shift_x
        y1 += shift_y
        y2 += shift_y

        # 1.3 pad size
        x1_pad = max(0, -x1)
        x2_pad = max(x2 - img_w + 1, 0)
        y1_pad = max(0, -y1)
        y2_pad = max(y2 - img_h + 1, 0)

        # 1.4 crop image
        img_crop = img[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]

        # 1.5 pad image
        img_crop_padded = cv2.copyMakeBorder(img_crop, y1_pad, y2_pad, x1_pad,
                                             x2_pad, cv2.BORDER_REPLICATE)

        # 2. Resize image and padding mask
        assert y2 - y1 == crop_size
        assert x2 - x1 == crop_size
        resize_factor = output_size / crop_size
        img_crop_padded = cv2.resize(img_crop_padded,
                                     (output_size, output_size))

        # the new box of cropped area
        crop_area_bbox = np.array([x1, y1, x2 - x1, y2 - y1], dtype=float)

        return img_crop_padded, crop_area_bbox, resize_factor

    def generate_box(self, bbox_gt: np.ndarray, crop_area_bbox: np.ndarray,
                     resize_factor: np.ndarray) -> np.ndarray:
        """Transform the box coordinates from the original image coordinates to
        the coordinates of the resized cropped image. The center of cropped
        image may be not jittered bbox since the operation of moving box inside
        image.

        Args:
            bbox_gt (np.ndarray): of shape (4, ) in [x1, y1, x2, y2] format.
            crop_area_bbox (np.ndarray): of shape (4, ) in [x1, y1, w, h]
                format.
            resize_factor (float): the ratio of original image scale to cropped
                image scale.
            output_size (float): the size of output image.
            normalize (bool): whether to normalize the output box.
                Default to True.

        Returns:
            np.ndarray: generated box of shape (4, ) in [x1, y1, x2, y2]
                format.
        """
        bbox_out = bbox_gt.copy()
        # The coordinate origin of `bbox_out` is the top left corner of
        # `crop_area_bbox`.
        bbox_out[0:4:2] -= crop_area_bbox[0]
        bbox_out[1:4:2] -= crop_area_bbox[1]

        bbox_out *= resize_factor

        return bbox_out

    def transform(self, results: dict) -> dict:
        """Call function. Crop image like DiMP did.

        Args:
            results (dict): Dict from :obj:`mmtrack.dataset.BaseSOTDataset`.

        Returns:
            dict: Dict that contains cropped images and
                corresponding ground truth boxes.
        """
        gt_bbox = results['gt_bboxes'][0]
        jittered_bboxes = results['jittered_bboxes'][0]
        crop_img, crop_area_bbox, resize_factor = self.crop_like_dimp(
            results['img'], jittered_bboxes, self.crop_size_factor,
            self.output_size)

        generated_bbox = self.generate_box(gt_bbox, crop_area_bbox,
                                           resize_factor)

        results['img'] = crop_img
        if 'img_shape' in results:
            results['img_shape'] = crop_img.shape
        results['gt_bboxes'] = generated_bbox[None]

        return results


@TRANSFORMS.register_module()
class SeqBboxJitter(BaseTransform):
    """Bounding box jitter augmentation. The jittered bboxes are used for
    subsequent image cropping, like `SeqCropLikeStark`.

    Required Keys:

    - gt_bboxes
    - gt_bboxes_labels (optional)
    - gt_instances_id (optional)
    - gt_masks (optional)
    - gt_seg_map (optional)
    - gt_ignore_flags (optional)
    - img
    - img_shape (optional)

    Added Keys:

    - jittered_bboxes

    Args:
        scale_jitter_factor (list[int | float]): contains the factor of scale
            jitter.
        center_jitter_factor (list[int | float]): contains the factor of center
            jitter.
        crop_size_factor (list[int | float]): contains the ratio of crop size
            to bbox size.
    """

    def __init__(self, scale_jitter_factor: List[Union[int, float]],
                 center_jitter_factor: List[Union[int, float]],
                 crop_size_factor: List[Union[int, float]]):
        self.scale_jitter_factor = scale_jitter_factor
        self.center_jitter_factor = center_jitter_factor
        self.crop_size_factor = crop_size_factor

    def transform(self, results: Dict[str, List]) -> Optional[dict]:
        """The transform function.

        Args:
            results (Dict[str, List]): Dict of list from
                :obj:`mmtrack.datasets.BaseSOTDataset`.

        Returns:
            Optional[dict]: Dict of list that contains augmented images. If
                getting invalid cropped image, return None.
        """
        gt_bboxes = results['gt_bboxes']
        jittered_bboxes = []
        for i, gt_bbox in enumerate(gt_bboxes):
            x1, y1, x2, y2 = np.split(gt_bbox.squeeze(), 4, axis=-1)
            bbox_w, bbox_h = x2 - x1, y2 - y1
            gt_bbox_cxcywh = np.concatenate(
                [x1 + bbox_w / 2., y1 + bbox_h / 2., bbox_w, bbox_h], axis=-1)

            crop_img_size = -1
            # avoid croped image size too small.
            count = 0
            while crop_img_size < 1:
                count += 1
                if count > 100:
                    print_log(
                        f'-------- bbox {gt_bbox_cxcywh} is invalid -------')
                    return None
                jittered_wh = gt_bbox_cxcywh[2:4] * np.exp(
                    np.random.randn(2) * self.scale_jitter_factor[i])
                crop_img_size = np.ceil(
                    np.sqrt(jittered_wh.prod()) * self.crop_size_factor[i])

            max_offset = np.sqrt(
                jittered_wh.prod()) * self.center_jitter_factor[i]
            jittered_center = gt_bbox_cxcywh[0:2] + max_offset * (
                np.random.rand(2) - 0.5)

            jittered_bbox = np.concatenate(
                (jittered_center - 0.5 * jittered_wh,
                 jittered_center + 0.5 * jittered_wh),
                axis=-1)
            jittered_bboxes.append(jittered_bbox[None])

        results['jittered_bboxes'] = jittered_bboxes
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'scale_jitter_factor={self.scale_jitter_factor}, '
        repr_str += f'center_jitter_factor={self.center_jitter_factor}, '
        repr_str += f'crop_size_factor={self.crop_size_factor})'
        return repr_str


@TRANSFORMS.register_module()
class BrightnessAug(BaseTransform):
    """Brightness augmention for images.

    Required Keys:

    - gt_bboxes
    - gt_bboxes_labels (optional)
    - gt_instances_id (optional)
    - gt_masks (optional)
    - gt_seg_map (optional)
    - gt_ignore_flags (optional)
    - img
    - img_shape (optional)

    Modified Keys:

    - img

    Args:
        jitter_range (float): The range of brightness jitter.
            Defaults to 0..
    """

    def __init__(self, jitter_range: float = 0.):
        self.jitter_range = jitter_range

    @cache_randomness
    def _random_brightness_factor(self) -> float:
        """Generate the factor of brightness randomly.

        Returns:
            float: The factor of brightness.
        """

        brightness_factor = np.random.uniform(
            max(0, 1 - self.jitter_range), 1 + self.jitter_range)
        return brightness_factor

    def transform(self, results: dict) -> dict:
        """The transform function.

        For each dict in results, perform brightness augmention for image in
        the dict.

        Args:
            results (dict): list of dict from :obj:`mmengine.BaseDataset`.
        Returns:
            dict: Dict that contains augmented image.
        """
        brightness_factor = self._random_brightness_factor()
        image = np.dot(results['img'], brightness_factor).clip(0, 255.0)
        results['img'] = image

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'jitter_range={self.jitter_range})'
        return repr_str


@TRANSFORMS.register_module()
class GrayAug(BaseTransform):
    """Gray augmention for images.

    Required Keys:

    - gt_bboxes
    - gt_bboxes_labels (optional)
    - gt_instances_id (optional)
    - gt_masks (optional)
    - gt_seg_map (optional)
    - gt_ignore_flags (optional)
    - img
    - img_shape (optional)

    Modified Keys:

    - img

    Args:
        prob (float): The probability to perform gray augmention.
            Defaults to 0..
    """

    def __init__(self, prob: float = 0.):
        self.prob = prob

    @cache_randomness
    def _random_gray(self) -> bool:
        """Whether to convert the original image to gray image.

        Returns:
            bool: Whether to convert the original image to gray image.
        """
        if self.prob > np.random.random():
            convert2gray = True
        else:
            convert2gray = False
        return convert2gray

    def transform(self, results: dict) -> dict:
        """The transform function.

        For each dict in results, perform gray augmention for image in the
        dict.

        Args:
            results (list[dict]): List of dict from
                :obj:`mmengine.BaseDataset`.

        Returns:
            dict: Dict that contains augmented gray image.
        """
        if self._random_gray():
            grayed = cv2.cvtColor(results['img'], cv2.COLOR_BGR2GRAY)
            image = cv2.cvtColor(grayed, cv2.COLOR_GRAY2BGR)
            results['img'] = image

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'prob={self.prob})'
        return repr_str


@TRANSFORMS.register_module()
class SeqShiftScaleAug(BaseTransform):
    """Shift and rescale images and bounding boxes.

    Required Keys:

    - gt_bboxes
    - gt_bboxes_labels (optional)
    - gt_instances_id (optional)
    - gt_masks (optional)
    - gt_seg_map (optional)
    - gt_ignore_flags (optional)
    - img
    - img_shape (optional)

    Modified Keys:

    - gt_bboxes
    - img
    - img_shape (optional)

    Args:
        target_size (list[int]): list of int denoting exemplar size and search
            size, respectively. Defaults to [127, 255].
        shift (list[int]): list of int denoting the max shift offset. Defaults
            to [4, 64].
        scale (list[float]): list of float denoting the max rescale factor.
            Defaults to [0.05, 0.18].
    """

    def __init__(self,
                 target_size: List[int] = [127, 255],
                 shift: List[int] = [4, 64],
                 scale: List[float] = [0.05, 0.18]):
        self.target_size = target_size
        self.shift = shift
        self.scale = scale

    def _shift_scale_aug(self, image: np.ndarray, bbox: np.ndarray,
                         target_size: int, shift: int,
                         scale: float) -> Tuple[np.ndarray, np.ndarray]:
        """Shift and rescale an image and corresponding bounding box.

        Args:
            image (np.ndarray): of shape (H, W, 3). Typically H and W equal to
                511.
            bbox (np.ndarray): of shape (4, ) in [x1, y1, x2, y2] format.
            target_size (int): Exemplar size or search size.
            shift (int): The max shift offset.
            scale (float): The max rescale factor.

        Returns:
            tuple(np.ndarray, np.ndarray): The first element is of shape
            (target_size, target_size, 3), and the second element is the
            corresponding ground truth box in [x1, y1, x2, y2] format.
        """
        img_h, img_w = image.shape[:2]

        scale_x = (2 * np.random.random() - 1) * scale + 1
        scale_y = (2 * np.random.random() - 1) * scale + 1
        scale_x = min(scale_x, float(img_w) / target_size)
        scale_y = min(scale_y, float(img_h) / target_size)
        crop_region = np.array([
            img_w // 2 - 0.5 * scale_x * target_size,
            img_h // 2 - 0.5 * scale_y * target_size,
            img_w // 2 + 0.5 * scale_x * target_size,
            img_h // 2 + 0.5 * scale_y * target_size
        ])

        shift_x = (2 * np.random.random() - 1) * shift
        shift_y = (2 * np.random.random() - 1) * shift
        shift_x = max(-crop_region[0], min(img_w - crop_region[2], shift_x))
        shift_y = max(-crop_region[1], min(img_h - crop_region[3], shift_y))
        shift = np.array([shift_x, shift_y, shift_x, shift_y])
        crop_region += shift

        crop_img = crop_image(image, crop_region, target_size)
        bbox -= np.array(
            [crop_region[0], crop_region[1], crop_region[0], crop_region[1]])
        bbox /= np.array([scale_x, scale_y, scale_x, scale_y],
                         dtype=np.float32)
        return crop_img, bbox

    def transform(self, results: dict) -> dict:
        """The transform function.

        For each dict in results, shift and rescale the image and the
        bounding box in the dict.

        Args:
            results (dict(list)): Dict of list that from
                :obj:`mmtrack.dataset.BaseSOTDataset`.

        Returns:
            dict(list): List of dict that contains cropped image and
            corresponding ground truth box.
        """
        imgs = results['img']
        gt_bboxes = results['gt_bboxes']
        new_imgs = []
        new_gt_bboxes = []
        for i, (img, gt_bbox) in enumerate(zip(imgs, gt_bboxes)):
            crop_img, crop_bbox = self._shift_scale_aug(
                img, gt_bbox.squeeze(), self.target_size[i], self.shift[i],
                self.scale[i])
            crop_bbox = crop_bbox[None]
            new_gt_bboxes.append(crop_bbox)
            new_imgs.append(crop_img)
            if 'img_shape' in results:
                results['img_shape'][i] = crop_img.shape
        results['img'] = new_imgs
        results['gt_bboxes'] = new_gt_bboxes
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(target_size={self.target_size}, '
        repr_str += f'shift={self.shift}, '
        repr_str += f'scale={self.scale})'
        return repr_str


@TRANSFORMS.register_module()
class SeqColorAug(BaseTransform):
    """Color augmention for images.

    Required Keys:

    - gt_bboxes
    - gt_bboxes_labels (optional)
    - gt_instances_id (optional)
    - gt_masks (optional)
    - gt_seg_map (optional)
    - gt_ignore_flags (optional)
    - img
    - img_shape (optional)

    Modified Keys:

    - img

    Args:
        prob (list[float]): The probability to perform color augmention for
            each image. Defaults to [1.0, 1.0].
        rgb_var (list[list]]): The values of color augmentaion. Defaults to
            [[-0.55919361, 0.98062831, -0.41940627],
            [1.72091413, 0.19879334, -1.82968581],
            [4.64467907, 4.73710203, 4.88324118]].
    """

    def __init__(self,
                 prob: List[float] = [1.0, 1.0],
                 rgb_var: List[List] = [[-0.55919361, 0.98062831, -0.41940627],
                                        [1.72091413, 0.19879334, -1.82968581],
                                        [4.64467907, 4.73710203, 4.88324118]]):
        self.prob = prob
        self.rgb_var = np.array(rgb_var, dtype=np.float32)

    def transform(self, results: dict) -> dict:
        """The transform function.

        For each dict in results, perform color augmention for image in the
        dict.

        Args:
            results (dict[list]): Dict of list that from
                :obj:`mmengine.BaseDataset`.

        Returns:
            dict[list]: Dict of list that contains augmented color image.
        """
        imgs = results['img']
        new_imgs = []
        for i, img in enumerate(imgs):
            if self.prob[i] > np.random.random():
                offset = np.dot(self.rgb_var, np.random.randn(3, 1))
                # bgr to rgb
                offset = offset[::-1]
                offset = offset.reshape(3)
                img = (img - offset).astype(np.float32)
            new_imgs.append(img)

        results['img'] = new_imgs
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'rgb_var={self.rgb_var})'
        return repr_str


@TRANSFORMS.register_module()
class SeqBlurAug(BaseTransform):
    """Blur augmention for images.

    Required Keys:

    - gt_bboxes
    - gt_bboxes_labels (optional)
    - gt_instances_id (optional)
    - gt_masks (optional)
    - gt_seg_map (optional)
    - gt_ignore_flags (optional)
    - img
    - img_shape (optional)

    Modified Keys:

    - img

    Args:
        prob (list[float]): The probability to perform blur augmention for
            each image. Defaults to [0.0, 0.2].
    """

    def __init__(self, prob: List[float] = [0.0, 0.2]):
        self.prob = prob

    def transform(self, results: dict) -> dict:
        """The transform function.

        For each dict in results, perform blur augmention for image in the
        dict.

        Args:
            results (dict[list]): Dict of list that from
                :obj:`mmtrack.CocoVideoDataset`.

        Returns:
            dict[list]: Dict of list that contains augmented blur image.
        """
        imgs = results['img']
        new_imgs = []
        for i, img in enumerate(imgs):
            if self.prob[i] > np.random.random():
                sizes = np.arange(5, 46, 2)
                size = np.random.choice(sizes)
                kernel = np.zeros((size, size))
                c = int(size / 2)
                wx = np.random.random()
                kernel[:, c] += 1. / size * wx
                kernel[c, :] += 1. / size * (1 - wx)
                img = cv2.filter2D(img, -1, kernel)
            new_imgs.append(img)

        results['img'] = new_imgs
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'prob={self.prob})'
        return repr_str


@TRANSFORMS.register_module()
class RandomCrop(MMDET_RandomCrop):
    """Random crop the image & bboxes & masks.

    We have rewritten a part of this function to facilitate the same processing
    of the `gt_instances_id` attribute during image clipping.For details of the
    function, see mmdetection.

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_bboxes_labels (np.int64) (optional)
    - gt_instances_id (np.int64) (optional)
    - gt_masks (BitmapMasks | PolygonMasks) (optional)
    - gt_ignore_flags (np.bool) (optional)
    - gt_seg_map (np.uint8) (optional)

    Modified Keys:

    - img
    - img_shape
    - gt_bboxes (optional)
    - gt_bboxes_labels (optional)
    - gt_instances_id (optional)
    - gt_masks (optional)
    - gt_ignore_flags (optional)
    - gt_seg_map (optional)

    Added Keys:

    - homography_matrix
    """

    def _crop_data(self, results: dict, crop_size: Tuple[int, int],
                   allow_negative_crop: bool) -> Union[dict, None]:
        """Function to randomly crop images, bounding boxes, masks, semantic
        segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.
            crop_size (Tuple[int, int]): Expected absolute size after
                cropping, (h, w).
            allow_negative_crop (bool): Whether to allow a crop that does not
                contain any bbox area.

        Returns:
            results (Union[dict, None]): Randomly cropped results, 'img_shape'
                key in result dict is updated according to crop size. None will
                be returned when there is no valid bbox after cropping.
        """
        assert crop_size[0] > 0 and crop_size[1] > 0
        img = results['img']
        margin_h = max(img.shape[0] - crop_size[0], 0)
        margin_w = max(img.shape[1] - crop_size[1], 0)
        offset_h, offset_w = self._rand_offset((margin_h, margin_w))
        crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

        # Record the homography matrix for the RandomCrop
        homography_matrix = np.array(
            [[1, 0, -offset_w], [0, 1, -offset_h], [0, 0, 1]],
            dtype=np.float32)
        if results.get('homography_matrix', None) is None:
            results['homography_matrix'] = homography_matrix
        else:
            results['homography_matrix'] = homography_matrix @ results[
                'homography_matrix']

        # crop the image
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        img_shape = img.shape
        results['img'] = img
        results['img_shape'] = img_shape

        # crop bboxes accordingly and clip to the image boundary
        if results.get('gt_bboxes', None) is not None:
            bboxes = results['gt_bboxes']
            bboxes.translate_([-offset_w, -offset_h])
            if self.bbox_clip_border:
                bboxes.clip_(img_shape[:2])
            valid_inds = bboxes.is_inside(img_shape[:2]).numpy()
            # If the crop does not contain any gt-bbox area and
            # allow_negative_crop is False, skip this image.
            if (not valid_inds.any() and not allow_negative_crop):
                return None

            results['gt_bboxes'] = bboxes[valid_inds]

            if results.get('gt_ignore_flags', None) is not None:
                results['gt_ignore_flags'] = \
                    results['gt_ignore_flags'][valid_inds]

            if results.get('gt_bboxes_labels', None) is not None:
                results['gt_bboxes_labels'] = \
                    results['gt_bboxes_labels'][valid_inds]

            if results.get('gt_instances_id', None) is not None:
                results['gt_instances_id'] = \
                    results['gt_instances_id'][valid_inds]

            if results.get('gt_masks', None) is not None:
                results['gt_masks'] = results['gt_masks'][
                    valid_inds.nonzero()[0]].crop(
                        np.asarray([crop_x1, crop_y1, crop_x2, crop_y2]))
                if self.recompute_bbox:
                    results['gt_bboxes'] = results['gt_masks'].get_bboxes(
                        type(results['gt_bboxes']))

        # crop semantic seg
        if results.get('gt_seg_map', None) is not None:
            results['gt_seg_map'] = results['gt_seg_map'][crop_y1:crop_y2,
                                                          crop_x1:crop_x2]

        return results
