# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import mmcv
import numpy as np
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import Normalize, Pad, RandomFlip, Resize

from mmtrack.core import crop_image


@PIPELINES.register_module()
class SeqCropLikeSiamFC(object):
    """Crop images as SiamFC did.

    The way of cropping an image is proposed in
    "Fully-Convolutional Siamese Networks for Object Tracking."
    `SiamFC <https://arxiv.org/abs/1606.09549>`_.

    Args:
        context_amount (float): The context amount around a bounding box.
            Defaults to 0.5.
        exemplar_size (int): Exemplar size. Defaults to 127.
        crop_size (int): Crop size. Defaults to 511.
    """

    def __init__(self, context_amount=0.5, exemplar_size=127, crop_size=511):
        self.context_amount = context_amount
        self.exemplar_size = exemplar_size
        self.crop_size = crop_size

    def crop_like_SiamFC(self,
                         image,
                         bbox,
                         context_amount=0.5,
                         exemplar_size=127,
                         crop_size=511):
        """Crop an image as SiamFC did.

        Args:
            image (ndarray): of shape (H, W, 3).
            bbox (ndarray): of shape (4, ) in [x1, y1, x2, y2] format.
            context_amount (float): The context amount around a bounding box.
                Defaults to 0.5.
            exemplar_size (int): Exemplar size. Defaults to 127.
            crop_size (int): Crop size. Defaults to 511.

        Returns:
            ndarray: The cropped image of shape (crop_size, crop_size, 3).
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
        d_search = (crop_size - exemplar_size) / 2
        pad = d_search / z_scale
        x_size = z_size + 2 * pad
        x_bbox = np.array([
            bbox[0] - 0.5 * x_size, bbox[1] - 0.5 * x_size,
            bbox[0] + 0.5 * x_size, bbox[1] + 0.5 * x_size
        ])

        x_crop_img = crop_image(image, x_bbox, crop_size, padding)
        return x_crop_img

    def generate_box(self, image, gt_bbox, context_amount, exemplar_size):
        """Generate box based on cropped image.

        Args:
            image (ndarray): The cropped image of shape
                (self.crop_size, self.crop_size, 3).
            gt_bbox (ndarray): of shape (4, ) in [x1, y1, x2, y2] format.
            context_amount (float): The context amount around a bounding box.
            exemplar_size (int): Exemplar size. Defaults to 127.

        Returns:
            ndarray: Generated box of shape (4, ) in [x1, y1, x2, y2] format.
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

    def __call__(self, results):
        """Call function.

        For each dict in results, crop image like SiamFC did.

        Args:
            results (list[dict]): List of dict that from
                :obj:`mmtrack.CocoVideoDataset`.

        Returns:
            list[dict]: List of dict that contains cropped image and
            corresponding ground truth box.
        """
        outs = []
        for _results in results:
            image = _results['img']
            gt_bbox = _results[_results.get('bbox_fields', [])[0]][0]

            crop_img = self.crop_like_SiamFC(image, gt_bbox,
                                             self.context_amount,
                                             self.exemplar_size,
                                             self.crop_size)
            generated_bbox = self.generate_box(crop_img, gt_bbox,
                                               self.context_amount,
                                               self.exemplar_size)
            generated_bbox = generated_bbox[None]

            _results['img'] = crop_img
            if 'img_shape' in _results:
                _results['img_shape'] = crop_img.shape
            _results[_results.get('bbox_fields', [])[0]] = generated_bbox

            outs.append(_results)
        return outs


@PIPELINES.register_module()
class SeqGrayAug(object):
    """Gray augmention for images.

    Args:
        prob (list[float]): The probability to perform gray augmention for
            each image. Defaults to 0..
    """

    def __init__(self, prob=0.):
        self.prob = prob

    def __call__(self, results):
        """Call function.

        For each dict in results, perform gray augmention for image in the
        dict.

        Args:
            results (list[dict]): List of dict that from
                :obj:`mmtrack.CocoVideoDataset`.

        Returns:
            list[dict]: List of dict that contains augmented gray image.
        """
        outs = []
        gray_prob = np.random.random()
        for _results in results:
            if self.prob > gray_prob:
                grayed = cv2.cvtColor(_results['img'], cv2.COLOR_BGR2GRAY)
                image = cv2.cvtColor(grayed, cv2.COLOR_GRAY2BGR)
                _results['img'] = image

            outs.append(_results)
        return outs


@PIPELINES.register_module()
class SeqShiftScaleAug(object):
    """Shift and rescale images and bounding boxes.

    Args:
        target_size (list[int]): list of int denoting exemplar size and search
            size, respectively. Defaults to [127, 255].
        shift (list[int]): list of int denoting the max shift offset. Defaults
            to [4, 64].
        scale (list[float]): list of float denoting the max rescale factor.
            Defaults to [0.05, 0.18].
    """

    def __init__(self,
                 target_size=[127, 255],
                 shift=[4, 64],
                 scale=[0.05, 0.18]):
        self.target_size = target_size
        self.shift = shift
        self.scale = scale

    def _shift_scale_aug(self, image, bbox, target_size, shift, scale):
        """Shift and rescale an image and corresponding bounding box.

        Args:
            image (ndarray): of shape (H, W, 3). Typically H and W equal to
                511.
            bbox (ndarray): of shape (4, ) in [x1, y1, x2, y2] format.
            target_size (int): Exemplar size or search size.
            shift (int): The max shift offset.
            scale (float): The max rescale factor.

        Returns:
            tuple(crop_img, bbox): crop_img is a ndarray of shape
            (target_size, target_size, 3), bbox is the corresponding ground
            truth box in [x1, y1, x2, y2] format.
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

    def __call__(self, results):
        """Call function.

        For each dict in results, shift and rescale the image and the bounding
        box in the dict.

        Args:
            results (list[dict]): List of dict that from
                :obj:`mmtrack.CocoVideoDataset`.

        Returns:
            list[dict]: List of dict that contains cropped image and
            corresponding ground truth box.
        """
        outs = []
        for i, _results in enumerate(results):
            image = _results['img']
            gt_bbox = _results[_results.get('bbox_fields', [])[0]][0]

            crop_img, crop_bbox = self._shift_scale_aug(
                image, gt_bbox, self.target_size[i], self.shift[i],
                self.scale[i])
            crop_bbox = crop_bbox[None]

            _results['img'] = crop_img
            if 'img_shape' in _results:
                _results['img_shape'] = crop_img.shape
            _results[_results.get('bbox_fields', [])[0]] = crop_bbox
            outs.append(_results)
        return outs


@PIPELINES.register_module()
class SeqColorAug(object):
    """Color augmention for images.

    Args:
        prob (list[float]): The probability to perform color augmention for
            each image. Defaults to [1.0, 1.0].
        rgb_var (list[list]]): The values of color augmentaion. Defaults to
            [[-0.55919361, 0.98062831, -0.41940627],
            [1.72091413, 0.19879334, -1.82968581],
            [4.64467907, 4.73710203, 4.88324118]].
    """

    def __init__(self,
                 prob=[1.0, 1.0],
                 rgb_var=[[-0.55919361, 0.98062831, -0.41940627],
                          [1.72091413, 0.19879334, -1.82968581],
                          [4.64467907, 4.73710203, 4.88324118]]):
        self.prob = prob
        self.rgb_var = np.array(rgb_var, dtype=np.float32)

    def __call__(self, results):
        """Call function.

        For each dict in results, perform color augmention for image in the
        dict.

        Args:
            results (list[dict]): List of dict that from
                :obj:`mmtrack.CocoVideoDataset`.

        Returns:
            list[dict]: List of dict that contains augmented color image.
        """
        outs = []
        for i, _results in enumerate(results):
            image = _results['img']

            if self.prob[i] > np.random.random():
                offset = np.dot(self.rgb_var, np.random.randn(3, 1))
                # bgr to rgb
                offset = offset[::-1]
                offset = offset.reshape(3)
                image = (image - offset).astype(np.float32)

            _results['img'] = image
            outs.append(_results)
        return outs


@PIPELINES.register_module()
class SeqBlurAug(object):
    """Blur augmention for images.

    Args:
        prob (list[float]): The probability to perform blur augmention for
            each image. Defaults to [0.0, 0.2].
    """

    def __init__(self, prob=[0.0, 0.2]):
        self.prob = prob

    def __call__(self, results):
        """Call function.

        For each dict in results, perform blur augmention for image in the
        dict.

        Args:
            results (list[dict]): List of dict that from
                :obj:`mmtrack.CocoVideoDataset`.

        Returns:
            list[dict]: List of dict that contains augmented blur image.
        """
        outs = []
        for i, _results in enumerate(results):
            image = _results['img']

            if self.prob[i] > np.random.random():
                sizes = np.arange(5, 46, 2)
                size = np.random.choice(sizes)
                kernel = np.zeros((size, size))
                c = int(size / 2)
                wx = np.random.random()
                kernel[:, c] += 1. / size * wx
                kernel[c, :] += 1. / size * (1 - wx)
                image = cv2.filter2D(image, -1, kernel)

            _results['img'] = image
            outs.append(_results)
        return outs


@PIPELINES.register_module()
class SeqResize(Resize):
    """Resize images.

    Please refer to `mmdet.datasets.pipelines.transforms.py:Resize` for
    detailed docstring.

    Args:
        share_params (bool): If True, share the resize parameters for all
            images. Defaults to True.
    """

    def __init__(self, share_params=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.share_params = share_params

    def __call__(self, results):
        """Call function.

        For each dict in results, call the call function of `Resize` to resize
        image and corresponding annotations.

        Args:
            results (list[dict]): List of dict that from
                :obj:`mmtrack.CocoVideoDataset`.

        Returns:
            list[dict]: List of dict that contains resized results,
            'img_shape', 'pad_shape', 'scale_factor', 'keep_ratio' keys
            are added into result dict.
        """
        outs, scale = [], None
        for i, _results in enumerate(results):
            if self.share_params and i > 0:
                _results['scale'] = scale
            _results = super().__call__(_results)
            if self.share_params and i == 0:
                scale = _results['scale']
            outs.append(_results)
        return outs


@PIPELINES.register_module()
class SeqNormalize(Normalize):
    """Normalize images.

    Please refer to `mmdet.datasets.pipelines.transforms.py:Normalize` for
    detailed docstring.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, results):
        """Call function.

        For each dict in results, call the call function of `Normalize` to
        normalize image.

        Args:
            results (list[dict]): List of dict that from
                :obj:`mmtrack.CocoVideoDataset`.

        Returns:
            list[dict]: List of dict that contains normalized results,
            'img_norm_cfg' key is added into result dict.
        """
        outs = []
        for _results in results:
            _results = super().__call__(_results)
            outs.append(_results)
        return outs


@PIPELINES.register_module()
class SeqRandomFlip(RandomFlip):
    """Randomly flip for images.

    Please refer to `mmdet.datasets.pipelines.transforms.py:RandomFlip` for
    detailed docstring.

    Args:
        share_params (bool): If True, share the flip parameters for all images.
            Defaults to True.
    """

    def __init__(self, share_params, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.share_params = share_params

    def __call__(self, results):
        """Call function.

        For each dict in results, call `RandomFlip` to randomly flip image.

        Args:
            results (list[dict]): List of dict that from
                :obj:`mmtrack.CocoVideoDataset`.

        Returns:
            list[dict]: List of dict that contains flipped results, 'flip',
            'flip_direction' keys are added into the dict.
        """
        if self.share_params:
            if isinstance(self.direction, list):
                # None means non-flip
                direction_list = self.direction + [None]
            else:
                # None means non-flip
                direction_list = [self.direction, None]

            if isinstance(self.flip_ratio, list):
                non_flip_ratio = 1 - sum(self.flip_ratio)
                flip_ratio_list = self.flip_ratio + [non_flip_ratio]
            else:
                non_flip_ratio = 1 - self.flip_ratio
                # exclude non-flip
                single_ratio = self.flip_ratio / (len(direction_list) - 1)
                flip_ratio_list = [single_ratio] * (len(direction_list) -
                                                    1) + [non_flip_ratio]

            cur_dir = np.random.choice(direction_list, p=flip_ratio_list)
            flip = cur_dir is not None
            flip_direction = cur_dir

            for _results in results:
                _results['flip'] = flip
                _results['flip_direction'] = flip_direction

        outs = []
        for _results in results:
            _results = super().__call__(_results)
            outs.append(_results)
        return outs


@PIPELINES.register_module()
class SeqPad(Pad):
    """Pad images.

    Please refer to `mmdet.datasets.pipelines.transforms.py:Pad` for detailed
    docstring.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, results):
        """Call function.

        For each dict in results, call the call function of `Pad` to pad image.

        Args:
            results (list[dict]): List of dict that from
                :obj:`mmtrack.CocoVideoDataset`.

        Returns:
            list[dict]: List of dict that contains padding results,
            'pad_shape', 'pad_fixed_size' and 'pad_size_divisor' keys are
            added into the dict.
        """
        outs = []
        for _results in results:
            _results = super().__call__(_results)
            outs.append(_results)
        return outs


@PIPELINES.register_module()
class SeqRandomCrop(object):
    """Sequentially random crop the images & bboxes & masks.

    The absolute `crop_size` is sampled based on `crop_type` and `image_size`,
    then the cropped results are generated.

    Args:
        crop_size (tuple): The relative ratio or absolute pixels of
            height and width.
        allow_negative_crop (bool, optional): Whether to allow a crop that does
            not contain any bbox area. Default False.
        share_params (bool, optional): Whether share the cropping parameters
            for the images.
        bbox_clip_border (bool, optional): Whether clip the objects outside
            the border of the image. Defaults to True.

    Note:
        - If the image is smaller than the absolute crop size, return the
            original image.
        - The keys for bboxes, labels and masks must be aligned. That is,
          `gt_bboxes` corresponds to `gt_labels` and `gt_masks`, and
          `gt_bboxes_ignore` corresponds to `gt_labels_ignore` and
          `gt_masks_ignore`.
        - If the crop does not contain any gt-bbox region and
          `allow_negative_crop` is set to False, skip this image.
    """

    def __init__(self,
                 crop_size,
                 allow_negative_crop=False,
                 share_params=False,
                 bbox_clip_border=False):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.allow_negative_crop = allow_negative_crop
        self.share_params = share_params
        self.bbox_clip_border = bbox_clip_border
        # The key correspondence from bboxes to labels and masks.
        self.bbox2label = {
            'gt_bboxes': ['gt_labels', 'gt_instance_ids'],
            'gt_bboxes_ignore': ['gt_labels_ignore', 'gt_instance_ids_ignore']
        }
        self.bbox2mask = {
            'gt_bboxes': 'gt_masks',
            'gt_bboxes_ignore': 'gt_masks_ignore'
        }

    def get_offsets(self, img):
        """Random generate the offsets for cropping."""
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        return offset_h, offset_w

    def random_crop(self, results, offsets=None):
        """Call function to randomly crop images, bounding boxes, masks,
        semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.
            offsets (tuple, optional): Pre-defined offsets for cropping.
                Default to None.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
            updated according to crop size.
        """

        for key in results.get('img_fields', ['img']):
            img = results[key]
            if offsets is not None:
                offset_h, offset_w = offsets
            else:
                offset_h, offset_w = self.get_offsets(img)
            results['img_info']['crop_offsets'] = (offset_h, offset_w)
            crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

            # crop the image
            img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
            img_shape = img.shape
            results[key] = img
        results['img_shape'] = img_shape

        # crop bboxes accordingly and clip to the image boundary
        for key in results.get('bbox_fields', []):
            # e.g. gt_bboxes and gt_bboxes_ignore
            bbox_offset = np.array([offset_w, offset_h, offset_w, offset_h],
                                   dtype=np.float32)
            bboxes = results[key] - bbox_offset
            if self.bbox_clip_border:
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            valid_inds = (bboxes[:, 2] > bboxes[:, 0]) & (
                bboxes[:, 3] > bboxes[:, 1])
            # If the crop does not contain any gt-bbox area and
            # self.allow_negative_crop is False, skip this image.
            if (key == 'gt_bboxes' and not valid_inds.any()
                    and not self.allow_negative_crop):
                return None
            results[key] = bboxes[valid_inds, :]
            # label fields. e.g. gt_labels and gt_labels_ignore
            label_keys = self.bbox2label.get(key)
            for label_key in label_keys:
                if label_key in results:
                    results[label_key] = results[label_key][valid_inds]

            # mask fields, e.g. gt_masks and gt_masks_ignore
            mask_key = self.bbox2mask.get(key)
            if mask_key in results:
                results[mask_key] = results[mask_key][
                    valid_inds.nonzero()[0]].crop(
                        np.asarray([crop_x1, crop_y1, crop_x2, crop_y2]))

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = results[key][crop_y1:crop_y2, crop_x1:crop_x2]
        return results

    def __call__(self, results):
        """Call function to sequentially randomly crop images, bounding boxes,
        masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
            updated according to crop size.
        """
        if self.share_params:
            offsets = self.get_offsets(results[0]['img'])
        else:
            offsets = None

        outs = []
        for _results in results:
            _results = self.random_crop(_results, offsets)
            if _results is None:
                return None
            outs.append(_results)

        return outs


@PIPELINES.register_module()
class SeqPhotoMetricDistortion(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 share_params=True,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.share_params = share_params
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def get_params(self):
        """Generate parameters."""
        params = dict()
        # delta
        if np.random.randint(2):
            params['delta'] = np.random.uniform(-self.brightness_delta,
                                                self.brightness_delta)
        else:
            params['delta'] = None
        # mode
        mode = np.random.randint(2)
        params['contrast_first'] = True if mode == 1 else 0
        # alpha
        if np.random.randint(2):
            params['alpha'] = np.random.uniform(self.contrast_lower,
                                                self.contrast_upper)
        else:
            params['alpha'] = None
        # saturation
        if np.random.randint(2):
            params['saturation'] = np.random.uniform(self.saturation_lower,
                                                     self.saturation_upper)
        else:
            params['saturation'] = None
        # hue
        if np.random.randint(2):
            params['hue'] = np.random.uniform(-self.hue_delta, self.hue_delta)
        else:
            params['hue'] = None
        # swap
        if np.random.randint(2):
            params['permutation'] = np.random.permutation(3)
        else:
            params['permutation'] = None
        return params

    def photo_metric_distortion(self, results, params=None):
        """Call function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.
            params (dict, optional): Pre-defined parameters. Default to None.

        Returns:
            dict: Result dict with images distorted.
        """
        if params is None:
            params = self.get_params()
        results['img_info']['color_jitter'] = params

        if 'img_fields' in results:
            assert results['img_fields'] == ['img'], \
                'Only single img_fields is allowed'
        img = results['img']
        assert img.dtype == np.float32, \
            'PhotoMetricDistortion needs the input image of dtype np.float32,'\
            ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
        # random brightness
        if params['delta'] is not None:
            img += params['delta']

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        if params['contrast_first']:
            if params['alpha'] is not None:
                img *= params['alpha']

        # convert color from BGR to HSV
        img = mmcv.bgr2hsv(img)

        # random saturation
        if params['saturation'] is not None:
            img[..., 1] *= params['saturation']

        # random hue
        if params['hue'] is not None:
            img[..., 0] += params['hue']
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        # convert color from HSV to BGR
        img = mmcv.hsv2bgr(img)

        # random contrast
        if not params['contrast_first']:
            if params['alpha'] is not None:
                img *= params['alpha']

        # randomly swap channels
        if params['permutation'] is not None:
            img = img[..., params['permutation']]

        results['img'] = img
        return results

    def __call__(self, results):
        """Call function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """
        if self.share_params:
            params = self.get_params()
        else:
            params = None

        outs = []
        for _results in results:
            _results = self.photo_metric_distortion(_results, params)
            outs.append(_results)

        return outs

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(\nbrightness_delta={self.brightness_delta},\n'
        repr_str += 'contrast_range='
        repr_str += f'{(self.contrast_lower, self.contrast_upper)},\n'
        repr_str += 'saturation_range='
        repr_str += f'{(self.saturation_lower, self.saturation_upper)},\n'
        repr_str += f'hue_delta={self.hue_delta})'
        return repr_str
