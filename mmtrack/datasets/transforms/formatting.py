# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np
from mmcv.transforms import BaseTransform

from mmtrack.registry import TRANSFORMS


@TRANSFORMS.register_module()
class ConcatSameTypeFrames(BaseTransform):
    """Concat the frames of the same type. We divide all the frames into two
    types: 'key' frames and 'reference' frames.

    The input dict is firstly convert from dict of list to list of dict. Then,
    we concat the first `num_key_frames` dicts to the first
    dict (key_frame_dict), and the rest of dicts are concated to the second
    dict (reference_frame_dict). Finally we merge two dicts with setting a
    prefix to all keys in the second dict.

    In SOT field, 'key' denotes search image and 'reference' denotes template
    image.

    Required Keys:

    - img (optional)
    - gt_bboxes (optional)
    - gt_labels (optional)
    - gt_instance_ids (optional)
    - gt_masks (optional)
    - proposals (optional)
    - file_name (optional)
    - frame_id (optional)
    - flip (optional)

    Modified Keys:

    - img (optional)
    - gt_bboxes (optional)
    - gt_labels (optional)
    - gt_instance_ids (optional)
    - gt_masks (optional)
    - proposals (optional)
    - file_name (optional)
    - frame_id (optional)
    - flip (optional)

    Added Keys:

    - f'{self.ref_prefix}_img' (optional)
    - f'{self.ref_prefix}_gt_bboxes' (optional)
    - f'{self.ref_prefix}_gt_labels' (optional)
    - f'{self.ref_prefix}_gt_instance_ids' (optional)
    - f'{self.ref_prefix}_gt_masks' (optional)
    - f'{self.ref_prefix}_proposals' (optional)
    - f'{self.ref_prefix}_file_name' (optional)
    - f'{self.ref_prefix}_frame_id' (optional)
    - f'{self.ref_prefix}_flip' (optional)

    Args:
        num_key_frames (int): the number of key frames.
        ref_prefix (str): The prefix of key added to the 'reference' frames.
            Defaults to 'ref'.
    """

    def __init__(self, num_key_frames, ref_prefix='ref'):
        self.num_key_frames = num_key_frames
        self.ref_prefix = ref_prefix

    def concat_one_mode_results(self, results):
        """Concatenate the results of the same mode."""
        out = dict()
        for i, result in enumerate(results):
            if 'img' in result:
                img = result['img']
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                if i == 0:
                    result['img'] = np.expand_dims(img, -1)
                else:
                    out['img'] = np.concatenate(
                        (out['img'], np.expand_dims(img, -1)), axis=-1)
            # TODO: remove the hard code
            for key in [
                    'file_name', 'img_id', 'frame_id', 'img_path', 'flip',
                    'flip_direction', 'gt_masks'
            ]:
                if key in result:
                    if i == 0:
                        result[key] = [result[key]]
                    else:
                        out[key].append(result[key])
            for key in [
                    'proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels',
                    'gt_instance_ids'
            ]:
                if key not in result:
                    continue
                value = result[key]
                if value.ndim == 1:
                    value = value[:, None]
                N = value.shape[0]
                value = np.concatenate((np.full(
                    (N, 1), i, dtype=value.dtype), value),
                                       axis=1)
                if i == 0:
                    result[key] = value
                else:
                    out[key] = np.concatenate((out[key], value), axis=0)
            if 'gt_semantic_seg' in result:
                if i == 0:
                    result['gt_semantic_seg'] = result['gt_semantic_seg'][...,
                                                                          None,
                                                                          None]
                else:
                    out['gt_semantic_seg'] = np.concatenate(
                        (out['gt_semantic_seg'],
                         result['gt_semantic_seg'][..., None, None]),
                        axis=-1)

            if 'padding_mask' in result:
                if i == 0:
                    result['padding_mask'] = np.expand_dims(
                        result['padding_mask'], 0)
                else:
                    out['padding_mask'] = np.concatenate(
                        (out['padding_mask'],
                         np.expand_dims(result['padding_mask'], 0)),
                        axis=0)

            if i == 0:
                out = result
        return out

    def transform(self, results):
        """Transform function.

        Args:
            results (dict): dict that contain keys such as 'img',
                'gt_masks','proposals', 'gt_bboxes',
                'gt_bboxes_ignore', 'gt_labels','gt_semantic_seg',
                'gt_instance_ids', 'padding_mask'.

        Returns:
            list[dict]: The first dict of outputs concats the dicts of 'key'
                information. The second dict of outputs concats the dicts of
                'reference' information.
        """
        assert (isinstance(results, dict)), 'results must be dict'
        # 1. Convert dict of list to list of dict
        # results['img'] == [img_1, img_2, img_3, ...]
        seq_len = len(results['img'])
        list_results = []
        for i in range(seq_len):
            _result = dict()
            for key in results.keys():
                _result[key] = results[key][i]
            list_results.append(_result)

        # 2. Concat results of key image and reference image separately.
        if self.num_key_frames > 1:
            key_results = self.concat_one_mode_results(
                list_results[:self.num_key_frames])
        else:
            # if single key, not expand the dim of variables
            key_results = list_results[0]
        reference_results = self.concat_one_mode_results(
            list_results[self.num_key_frames:])

        # 3. Set prefix for keys in the results of reference image.
        results = key_results
        for k, v in reference_results.items():
            results[f'{self.ref_prefix}_{k}'] = v
        return results


@TRANSFORMS.register_module()
class ConcatVideoReferences(ConcatSameTypeFrames):
    """Concat video references.

    There is only one key frame in the results and perhaps multiple reference
    frames in the results. We will concat all reference frames in this
    Transform.
    """

    def __init__(self, **kwargs):
        super(ConcatVideoReferences, self).__init__(num_key_frames=1, **kwargs)


@TRANSFORMS.register_module()
class CheckPadMaskValidity(object):
    """Check the validity of data. Generally, it's used in such case: The image
    padding masks generated in the image preprocess need to be downsampled, and
    then passed into Transformer model, like DETR. The computation in the
    subsequent Transformer model must make sure that the values of downsampled
    mask are not all zeros.

    Args:
        stride (int): the max stride of feature map.
    """

    def __init__(self, stride):
        self.stride = stride

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): Result dict contains the data to be checked.

        Returns:
            dict | None: If invalid, return None; otherwise, return original
                input.
        """
        for _results in results:
            assert 'padding_mask' in _results
            mask = _results['padding_mask'].copy().astype(np.float32)
            img_h, img_w = _results['img'].shape[:2]
            feat_h, feat_w = img_h // self.stride, img_w // self.stride
            downsample_mask = cv2.resize(
                mask, dsize=(feat_h, feat_w)).astype(bool)
            if (downsample_mask == 1).all():
                return None
        return results
