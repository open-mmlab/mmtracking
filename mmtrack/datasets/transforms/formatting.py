# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import cv2
import numpy as np
from mmcv.transforms import BaseTransform, to_tensor
from mmengine.data import InstanceData

from mmtrack.core import TrackDataSample
from mmtrack.registry import TRANSFORMS


# TODO: We may consider to merge ``ConcatSameTypeFrames`` into
# ``PackTrackDataSample``.
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
    - gt_bboxes_labels (optional)
    - gt_ignore_flags (optional)
    - gt_instance_ids (optional)
    - gt_masks (optional)
    - proposals (optional)
    - file_name (optional)
    - frame_id (optional)
    - flip (optional)

    Modified Keys:

    - img (optional)
    - gt_bboxes (optional)
    - gt_bboxes_labels (optional)
    - gt_ignore_flags (optional)
    - gt_instance_ids (optional)
    - gt_masks (optional)
    - proposals (optional)
    - file_name (optional)
    - frame_id (optional)
    - flip (optional)

    Added Keys:

    - f'{self.ref_prefix}_img' (optional)
    - f'{self.ref_prefix}_gt_bboxes' (optional)
    - f'{self.ref_prefix}_gt_bboxes_labels' (optional)
    - f'{self.ref_prefix}_gt_ignore_flags' (optional)
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
        meta_keys (Sequence[str]): Meta keys to be collected in
            ``data_sample.metainfo``. Defaults to None.
        default_meta_keys (tuple): Default meta keys. Defaults to ('filename',
            'ori_filename', 'ori_shape', 'img_shape', 'pad_shape',
            'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg',
            'frame_id', 'is_video_data').
    """

    def __init__(self,
                 num_key_frames: int,
                 ref_prefix: str = 'ref',
                 meta_keys: Optional[dict] = None,
                 default_meta_keys: dict = ('img_id', 'img_path', 'ori_shape',
                                            'img_shape', 'scale_factor',
                                            'flip', 'flip_direction',
                                            'frame_id', 'is_video_data')):
        self.num_key_frames = num_key_frames
        self.ref_prefix = ref_prefix
        self.meta_keys = default_meta_keys
        if meta_keys is not None:
            if isinstance(meta_keys, str):
                meta_keys = (meta_keys, )
            else:
                assert isinstance(meta_keys, tuple), \
                    'meta_keys must be str or tuple'
            self.meta_keys += meta_keys

    def concat_one_mode_results(self, results: dict) -> dict:
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
            for key in (self.meta_keys + ('gt_masks', )):
                if key in result:
                    if i == 0:
                        result[key] = [result[key]]
                    else:
                        out[key].append(result[key])
            for key in [
                    'proposals', 'gt_bboxes', 'gt_ignore_flags',
                    'gt_bboxes_labels', 'gt_instances_id'
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

    def transform(self, results: dict) -> dict:
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

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'num_key_frames={self.num_key_frames}, '
        repr_str += f'ref_prefix={self.ref_prefix}, '
        repr_str += f'meta_keys={self.meta_keys}, '
        repr_str += f'default_meta_keys={self.default_meta_keys})'
        return repr_str


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
class PackTrackInputs(BaseTransform):
    """Pack the inputs data for the video object detection / multi object
    tracking / single object tracking / video instance segmentation.

    Args:
        ref_prefix (str): The prefix of key added to the 'reference' frames.
            Defaults to 'ref'.
        meta_keys (Sequence[str]): Meta keys to be collected in
            ``data_sample.metainfo``. Defaults to None.
        default_meta_keys (tuple): Default meta keys. Defaults to ('filename',
            'ori_filename', 'ori_shape', 'img_shape', 'pad_shape',
            'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg',
            'frame_id', 'is_video_data').
    """
    mapping_table = {
        'gt_bboxes': 'bboxes',
        'gt_bboxes_labels': 'labels',
        'gt_masks': 'masks',
        'gt_instances_id': 'instances_id'
    }

    def __init__(self,
                 ref_prefix: str = 'ref',
                 meta_keys: Optional[dict] = None,
                 default_meta_keys: dict = ('img_id', 'img_path', 'ori_shape',
                                            'img_shape', 'scale_factor',
                                            'flip', 'flip_direction',
                                            'frame_id', 'is_video_data')):
        self.ref_prefix = ref_prefix
        self.meta_keys = default_meta_keys
        if meta_keys is not None:
            if isinstance(meta_keys, str):
                meta_keys = (meta_keys, )
            else:
                assert isinstance(meta_keys, tuple), \
                    'meta_keys must be str or tuple'
            self.meta_keys += meta_keys

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_sample' (obj:`TrackDataSample`): The annotation info of the
                sample.
        """
        packed_results = dict()
        packed_results['inputs'] = dict()

        # 1. Pack image of key frames
        if 'img' in results:
            img = results['img']
            if len(img.shape) <= 3:
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                # (H, W, C) -> (C, H, W)
                img = np.ascontiguousarray(img.transpose(2, 0, 1))
            else:
                # (H, W, C, N) -> (N, C, H, W)
                img = np.ascontiguousarray(img.transpose(3, 2, 0, 1))
            packed_results['inputs']['img'] = to_tensor(img)

        # 2. Pack image of reference frames.
        if f'{self.ref_prefix}_img' in results:
            ref_img = results[f'{self.ref_prefix}_img']
            ref_img = np.ascontiguousarray(ref_img.transpose(3, 2, 0, 1))
            packed_results['inputs'][f'{self.ref_prefix}_img'] = to_tensor(
                ref_img)

        data_sample = TrackDataSample()

        # 3. Pack data of key frames
        if 'gt_ignore_flags' in results:
            vaild_idx = results['gt_ignore_flags'] == 0
            ignore_idx = results['gt_ignore_flags'] == 1

        instance_data = InstanceData()
        ignore_instance_data = InstanceData()
        for key in self.mapping_table.keys():
            if key not in results:
                continue
            if key == 'gt_masks':
                if 'gt_ignore_flags' in results:
                    instance_data[
                        self.mapping_table[key]] = results[key][vaild_idx]
                    ignore_instance_data[
                        self.mapping_table[key]] = results[key][ignore_idx]
                else:
                    instance_data[self.mapping_table[key]] = results[key]
            else:
                if 'gt_ignore_flags' in results:
                    instance_data[self.mapping_table[key]] = to_tensor(
                        results[key][vaild_idx])
                    ignore_instance_data[self.mapping_table[key]] = to_tensor(
                        results[key][ignore_idx])
                else:
                    instance_data[self.mapping_table[key]] = to_tensor(
                        results[key])
        data_sample.gt_instances = instance_data
        data_sample.ignored_instances = ignore_instance_data

        if 'proposals' in results:
            data_sample.proposals = InstanceData('bboxes',
                                                 results['proposals'])

        # 4. Pack data of reference frames
        if f'{self.ref_prefix}_gt_ignore_flags' in results:
            vaild_idx = results[f'{self.ref_prefix}_gt_ignore_flags'][:,
                                                                      1] == 0
            ignore_idx = results[f'{self.ref_prefix}_gt_ignore_flags'][:,
                                                                       1] == 1

        ref_instance_data = InstanceData()
        ref_ignore_instance_data = InstanceData()
        for key in self.mapping_table.keys():
            ref_key = f'{self.ref_prefix}_{key}'
            if ref_key not in results:
                continue
            if ref_key == f'{self.ref_prefix}_gt_masks':
                if f'{self.ref_prefix}_gt_ignore_flags' in results:
                    ref_instance_data[
                        self.mapping_table[key]] = results[ref_key][vaild_idx]
                    ref_ignore_instance_data[
                        self.mapping_table[key]] = results[ref_key][ignore_idx]
                else:
                    ref_instance_data[
                        self.mapping_table[key]] = results[ref_key]
            else:
                if f'{self.ref_prefix}_gt_ignore_flags' in results:
                    ref_instance_data[self.mapping_table[key]] = to_tensor(
                        results[ref_key][vaild_idx])
                    ref_ignore_instance_data[
                        self.mapping_table[key]] = to_tensor(
                            results[ref_key][ignore_idx])
                else:
                    ref_instance_data[self.mapping_table[key]] = to_tensor(
                        results[ref_key])
        setattr(data_sample, f'{self.ref_prefix}_gt_instances',
                ref_instance_data)
        setattr(data_sample, f'{self.ref_prefix}_ignore_instance_data',
                ref_ignore_instance_data)

        if f'{self.ref_prefix}_proposals' in results:
            setattr(
                data_sample, f'{self.ref_prefix}_proposals',
                InstanceData('bboxes',
                             results[f'{self.ref_prefix}_proposals']))

        # 5. set metainfo
        img_meta = {}
        for key in self.meta_keys:
            # TODO: Wait until mmcv is modified and deleted.
            if (key == 'ori_shape' and 'ori_height' in results
                    and 'ori_width' in results):
                img_shape = (results['ori_height'], results['ori_width'])
                img_meta[key] = img_shape
            else:
                if key not in results:
                    continue
                img_meta[key] = results[key]
                img_meta[f'{self.ref_prefix}_{key}'] = results[
                    f'{self.ref_prefix}_{key}']
        data_sample.set_metainfo(img_meta)

        packed_results['data_sample'] = data_sample
        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(ref_prefix={self.ref_prefix}, '
        repr_str += f'meta_keys={self.meta_keys}, '
        repr_str += f'default_meta_keys={self.default_meta_keys})'
        return repr_str


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
