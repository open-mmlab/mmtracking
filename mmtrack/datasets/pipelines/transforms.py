import mmcv
import numpy as np
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import Normalize, Pad, RandomFlip, Resize


@PIPELINES.register_module()
class SeqResize(Resize):

    def __init__(self, share_params=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.share_params = share_params

    def __call__(self, results):
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, results):
        outs = []
        for _results in results:
            _results = super().__call__(_results)
            outs.append(_results)
        return outs


@PIPELINES.register_module()
class SeqRandomFlip(RandomFlip):

    def __init__(self, share_params, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.share_params = share_params

    def __call__(self, results):
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, results):
        outs = []
        for _results in results:
            _results = super().__call__(_results)
            outs.append(_results)
        return outs


@PIPELINES.register_module()
class SeqRandomCrop(object):

    def __init__(self,
                 crop_size,
                 allow_negative_crop=False,
                 share_params=False):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.allow_negative_crop = allow_negative_crop
        self.share_params = share_params
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
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        return offset_h, offset_w

    def call(self, results, offsets=None):
        """Call function to randomly crop images, bounding boxes, masks,
        semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

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
        if self.share_params:
            offsets = self.get_offsets(results[0]['img'])
        else:
            offsets = None

        outs = []
        for _results in results:
            _results = self.call(_results, offsets)
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

    def call(self, results, params=None):
        """Call function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.

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
        if self.share_params:
            params = self.get_params()
        else:
            params = None

        outs = []
        for _results in results:
            _results = self.call(_results, params)
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
