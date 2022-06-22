# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from mmcv.transforms import BaseTransform, to_tensor
from mmdet.core.mask import BitmapMasks
from mmengine.data import InstanceData

from mmtrack.core import ReIDDataSample, TrackDataSample
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
        meta_keys (Optional[Union[str, Tuple]], optional): Meta keys to be
            collected in ``data_sample.metainfo``. Defaults to None.
        default_meta_keys (tuple, optional): Default meta keys. Defaults to
            ('filename', 'ori_filename', 'ori_shape', 'img_shape', 'pad_shape',
            'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg',
            'frame_id', 'is_video_data').
    """

    def __init__(self,
                 num_key_frames: int,
                 ref_prefix: str = 'ref',
                 meta_keys: Optional[Union[str, Tuple]] = None,
                 default_meta_keys: tuple = ('img_id', 'img_path', 'ori_shape',
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
            dict: The elements without prefix ``self.ref_prefix`` concats the
                information of key frame. The elements with prefix
                ``self.ref_prefix`` concats the information of reference frame.
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

    For each value (``List`` type) in the input dict, we concat the first
    `num_key_frames` elements to the first dict with a new key, and the rest
    of elements are concated to the second dict with a new key.
    All the information of images are packed to ``inputs``.
    All the information except images are packed to ``data_samples``.

    Args:
        ref_prefix (str): The prefix of key added to the 'reference' frames.
            Defaults to 'ref'.
        num_key_frames (int): The number of key frames. Defaults to 1.
        num_template_frames (optional, int): The number of template frames. It
            is only used for training in SOT.
        pack_single_img (bool, optional): Whether to only pack single image. If
            True, pack the data as a list additionally. Defaults to False.
        meta_keys (Sequence[str]): Meta keys to be collected in
            ``data_sample.metainfo``. Defaults to None.
        default_meta_keys (tuple): Default meta keys. Defaults to ('img_id',
            'img_path', 'ori_shape', 'img_shape', 'scale_factor',
            'flip', 'flip_direction', 'frame_id', 'is_video_data',
            'video_id', 'video_length', 'instances').
    """
    mapping_table = {
        'gt_bboxes': 'bboxes',
        'gt_bboxes_labels': 'labels',
        'gt_masks': 'masks',
        'gt_instances_id': 'instances_id'
    }

    def __init__(self,
                 ref_prefix: str = 'ref',
                 num_key_frames: int = 1,
                 num_template_frames: Optional[int] = None,
                 pack_single_img: Optional[bool] = False,
                 meta_keys: Optional[dict] = None,
                 default_meta_keys: tuple = ('img_id', 'img_path', 'ori_shape',
                                             'img_shape', 'scale_factor',
                                             'flip', 'flip_direction',
                                             'frame_id', 'is_video_data',
                                             'video_id', 'video_length',
                                             'instances')):
        self.ref_prefix = ref_prefix
        # If ``num_template_frames`` is not None, this class is used in SOT.
        # In this case, we assign the value of ``num_template_frames`` to
        # ``self.num_key_frames`` for the consistency in the processing.
        self.num_key_frames = num_key_frames if num_template_frames is None \
            else num_template_frames
        self.meta_keys = default_meta_keys
        if meta_keys is not None:
            if isinstance(meta_keys, str):
                meta_keys = (meta_keys, )
            else:
                assert isinstance(meta_keys, tuple), \
                    'meta_keys must be str or tuple'
            self.meta_keys += meta_keys

        self.pack_single_img = pack_single_img

    def _cat_same_type_data(
            self,
            data: Union[List, int],
            return_ndarray: bool = True,
            axis: int = 0,
            stack: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Concatenate data with the same type.

        Args:
            data (Union[List, int]): Input data.
            return_ndarray (bool, optional): Whether to return ``np.ndarray``.
                Defaults to True.
            axis (int, optional): The axis that concatenating along. Defaults
                to 0.
            stack (bool, optional): Whether to stack all the data. If not,
                using the concatenating operation. Defaults to False.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The first element is the
                concatenated data of key frames, and the second element is the
                concatenated data of reference frames.
        """
        if self.pack_single_img:
            data = [data]
        key_data = data[:self.num_key_frames]
        ref_data = data[self.num_key_frames:] if len(
            data) > self.num_key_frames else None

        if return_ndarray:
            if stack:
                key_data = np.stack(key_data, axis=axis)
                if ref_data is not None:
                    ref_data = np.stack(ref_data, axis=axis)
            else:
                key_data = np.concatenate(key_data, axis=axis)
                if ref_data is not None:
                    ref_data = np.concatenate(ref_data, axis=axis)

        return key_data, ref_data

    def _get_img_idx_map(self, anns: List) -> Tuple[np.ndarray, np.ndarray]:
        """Get the index of images for the annotations. The multiple instances
        in one image need to be denoted the image index when concatenating
        multiple images.

        Args:
            anns (List): Input annotations.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The first element is the
                concatenated indexes of key frames, and the second element is
                the concatenated indexes of reference frames.
        """
        if self.pack_single_img:
            anns = [anns]
        key_img_idx_map = []
        for img_idx, ann in enumerate(anns[:self.num_key_frames]):
            key_img_idx_map.extend([img_idx] * len(ann))
        key_img_idx_map = np.array(key_img_idx_map, dtype=np.int32)
        if len(anns) > self.num_key_frames:
            ref_img_idx_map = []
            for img_idx, ann in enumerate(anns[self.num_key_frames:]):
                ref_img_idx_map.extend([img_idx] * len(ann))
            ref_img_idx_map = np.array(ref_img_idx_map, dtype=np.int32)
        else:
            ref_img_idx_map = None
        return key_img_idx_map, ref_img_idx_map

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (dict[Tensor]): The forward data of models.
            - 'data_sample' (obj:`TrackDataSample`): The annotation info of the
                sample.
        """
        packed_results = dict()
        packed_results['inputs'] = dict()

        # 1. Pack images
        if 'img' in results:
            imgs = results['img']
            key_imgs, ref_imgs = self._cat_same_type_data(imgs, stack=True)
            key_imgs = key_imgs.transpose(0, 3, 1, 2)
            packed_results['inputs']['img'] = to_tensor(key_imgs)

            if ref_imgs is not None:
                ref_imgs = ref_imgs.transpose(0, 3, 1, 2)
                packed_results['inputs'][f'{self.ref_prefix}_img'] = to_tensor(
                    ref_imgs)

        data_sample = TrackDataSample()

        # 2. Pack InstanceData
        if 'gt_ignore_flags' in results:
            gt_ignore_flags = results['gt_ignore_flags']
            (key_gt_ignore_flags,
             ref_gt_ignore_flags) = self._cat_same_type_data(gt_ignore_flags)
            key_valid_idx = key_gt_ignore_flags == 0
            if ref_gt_ignore_flags is not None:
                ref_valid_idx = ref_gt_ignore_flags == 0

        instance_data = InstanceData()
        ignore_instance_data = InstanceData()
        ref_instance_data = InstanceData()
        ref_ignore_instance_data = InstanceData()

        # Flag that whether have recorded the image index
        img_idx_map_flag = False
        for key in self.mapping_table.keys():
            if key not in results:
                continue
            if key == 'gt_masks':
                gt_masks = results[key]
                gt_masks_ndarray = [
                    mask.to_ndarray() for mask in gt_masks
                ] if isinstance(gt_masks, list) else gt_masks.to_ndarray()
                key_gt_masks, ref_gt_masks = self._cat_same_type_data(
                    gt_masks_ndarray)

                mapped_key = self.mapping_table[key]
                if 'gt_ignore_flags' in results:
                    instance_data[mapped_key] = BitmapMasks(
                        key_gt_masks[key_valid_idx], *key_gt_masks.shape[-2:])
                    ignore_instance_data[mapped_key] = BitmapMasks(
                        key_gt_masks[~key_valid_idx], *key_gt_masks.shape[-2:])

                    if ref_gt_masks is not None:
                        ref_instance_data[mapped_key] = BitmapMasks(
                            ref_gt_masks[ref_valid_idx],
                            *key_gt_masks.shape[-2:])
                        ref_ignore_instance_data[mapped_key] = BitmapMasks(
                            ref_gt_masks[~ref_valid_idx],
                            *key_gt_masks.shape[-2:])
                else:
                    instance_data[mapped_key] = BitmapMasks(
                        key_gt_masks, *key_gt_masks.shape[-2:])
                    if ref_gt_masks is not None:
                        ref_instance_data[mapped_key] = BitmapMasks(
                            ref_gt_masks, *ref_gt_masks.shape[-2:])

            else:
                anns = results[key]
                key_anns, ref_anns = self._cat_same_type_data(anns)

                if not img_idx_map_flag:
                    # The multiple instances in one image need to be
                    # denoted the image index when concatenating multiple
                    # images.
                    key_img_idx_map, ref_img_idx_map = self._get_img_idx_map(
                        anns)
                    img_idx_map_flag = True

                mapped_key = self.mapping_table[key]
                if 'gt_ignore_flags' in results:
                    instance_data[mapped_key] = to_tensor(
                        key_anns[key_valid_idx])
                    ignore_instance_data[mapped_key] = to_tensor(
                        key_anns[~key_valid_idx])
                    instance_data['map_instances_to_img_idx'] = to_tensor(
                        key_img_idx_map[key_valid_idx])
                    ignore_instance_data[
                        'map_instances_to_img_idx'] = to_tensor(
                            key_img_idx_map[~key_valid_idx])

                    if ref_anns is not None:
                        ref_instance_data[mapped_key] = to_tensor(
                            ref_anns[ref_valid_idx])
                        ref_ignore_instance_data[mapped_key] = to_tensor(
                            ref_anns[~ref_valid_idx])
                        ref_instance_data[
                            'map_instances_to_img_idx'] = to_tensor(
                                ref_img_idx_map[ref_valid_idx])
                        ref_ignore_instance_data[
                            'map_instances_to_img_idx'] = to_tensor(
                                ref_img_idx_map[~ref_valid_idx])
                else:
                    instance_data[mapped_key] = to_tensor(key_anns)
                    instance_data['map_instances_to_img_idx'] = to_tensor(
                        key_img_idx_map)
                    if ref_anns is not None:
                        ref_instance_data[mapped_key] = to_tensor(ref_anns)
                        ref_instance_data[
                            'map_instances_to_img_idx'] = to_tensor(
                                ref_img_idx_map)

        data_sample.gt_instances = instance_data
        data_sample.ignored_instances = ignore_instance_data
        setattr(data_sample, f'{self.ref_prefix}_gt_instances',
                ref_instance_data)
        setattr(data_sample, f'{self.ref_prefix}_ignored_instances',
                ref_ignore_instance_data)

        # 3. Pack metainfo
        new_img_metas = {}
        for key in self.meta_keys:
            # TODO: Wait until mmcv is modified and deleted.
            if (key == 'ori_shape' and 'ori_height' in results
                    and 'ori_width' in results):
                ori_height = results['ori_height']
                ori_width = results['ori_width']
                key_ori_height, ref_ori_height = self._cat_same_type_data(
                    ori_height, return_ndarray=False)
                key_ori_width, ref_ori_width = self._cat_same_type_data(
                    ori_width, return_ndarray=False)
                assert len(key_ori_height) == len(key_ori_width)
                # To compatible the interface of ``MMDet``, we don't use
                # the fotmat of list when the length of meta information is
                # equal to 1.
                if len(key_ori_height) > 1:
                    new_img_metas[key] = [
                        (h, w) for h, w in zip(key_ori_height, key_ori_width)
                    ]
                else:
                    new_img_metas[key] = (key_ori_height[0], key_ori_width[0])
                if ref_ori_height is not None and ref_ori_width is not None:
                    assert len(ref_ori_height) == len(ref_ori_width)
                    if len(ref_ori_height) > 1:
                        new_img_metas[f'{self.ref_prefix}_{key}'] = [
                            (h, w)
                            for h, w in zip(ref_ori_height, ref_ori_width)
                        ]
                    else:
                        new_img_metas[f'{self.ref_prefix}_{key}'] = (
                            ref_ori_height[0], ref_ori_width[0])
            else:
                if key not in results:
                    continue
                img_metas = results[key]
                key_img_metas, ref_img_metas = self._cat_same_type_data(
                    img_metas, return_ndarray=False)
                # To compatible the interface of ``MMDet``, we don't use
                # the fotmat of list when the length of meta information is
                # equal to 1.
                if len(key_img_metas) > 1:
                    new_img_metas[key] = key_img_metas
                else:
                    new_img_metas[key] = key_img_metas[0]
                if ref_img_metas is not None:
                    if len(ref_img_metas) > 1:
                        new_img_metas[
                            f'{self.ref_prefix}_{key}'] = ref_img_metas
                    else:
                        new_img_metas[
                            f'{self.ref_prefix}_{key}'] = ref_img_metas[0]

        data_sample.set_metainfo(new_img_metas)

        # 4. Pack some additional properties.
        if 'padding_mask' in results:
            # This property is used in ``STARK`` method in SOT.
            padding_mask = results['padding_mask']
            key_padding_mask, ref_padding_mask = self._cat_same_type_data(
                padding_mask, stack=True)
            data_sample.padding_mask = to_tensor(key_padding_mask)
            if ref_padding_mask is not None:
                setattr(data_sample, f'{self.ref_prefix}_padding_mask',
                        to_tensor(ref_padding_mask))

        packed_results['data_sample'] = data_sample
        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(ref_prefix={self.ref_prefix}, '
        repr_str += f'meta_keys={self.meta_keys}, '
        repr_str += f'default_meta_keys={self.default_meta_keys})'
        return repr_str


@TRANSFORMS.register_module()
class CheckPadMaskValidity(BaseTransform):
    """Check the validity of data. Generally, it's used in such case: The image
    padding masks generated in the image preprocess need to be downsampled, and
    then passed into Transformer model, like DETR. The computation in the
    subsequent Transformer model must make sure that the values of downsampled
    mask are not all zeros.

    Required Keys:

    - gt_bboxes (np.float32)
    - gt_bboxes_labels (np.int32)
    - gt_instances_id (np.int32)
    - gt_masks (BitmapMasks | PolygonMasks)
    - gt_seg_map (np.uint8)
    - gt_ignore_flags (np.bool)
    - img
    - img_shape (optional)
    - jittered_bboxes (optional)
    - padding_mask (np.bool)

    Args:
        stride (int): the max stride of feature map.
    """

    def __init__(self, stride: int):
        self.stride = stride

    def transform(self, results: dict) -> Optional[dict]:
        """The transform function.

        Args:
            results (dict): Result dict contains the data to be checked.

        Returns:
            Optional[dict]: If invalid, return None; otherwise, return original
                input.
        """
        assert 'padding_mask' in results
        masks = results['padding_mask']
        imgs = results['img']
        for mask, img in zip(masks, imgs):
            mask = mask.copy().astype(np.float32)
            img_h, img_w = img.shape[:2]
            feat_h, feat_w = img_h // self.stride, img_w // self.stride
            downsample_mask = cv2.resize(
                mask, dsize=(feat_h, feat_w)).astype(bool)
            if (downsample_mask == 1).all():
                return None
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'stride={self.stride})'
        return repr_str


@TRANSFORMS.register_module()
class PackReIDInputs(BaseTransform):
    """Pack the inputs data for the ReID.

    The ``meta_info`` item is always populated. The contents of the
    ``meta_info`` dictionary depends on ``meta_keys``. By default
    this includes:

        - ``img_path``: path to the image file.

        - ``ori_shape``: original shape of the image as a tuple (H, W).

        - ``img_shape``: shape of the image input to the network as a tuple
            (H, W). Note that images may be zero padded on the bottom/right
          if the batch tensor is larger than this shape.

        - ``scale``: scale of the image as a tuple (W, H).

        - ``scale_factor``: a float indicating the pre-processing scale.

        -  ``flip``: a boolean indicating if image flip transform was used.

        - ``flip_direction``: the flipping direction.

    Args:
        meta_keys (Sequence[str], optional): The meta keys to saved in the
            ``metainfo`` of the packed ``data_sample``.
    """
    default_meta_keys = ('img_path', 'ori_shape', 'img_shape', 'scale',
                         'scale_factor', 'flip', 'flip_direction')

    def __init__(self, meta_keys: Sequence[str] = ()) -> None:
        self.meta_keys = self.default_meta_keys
        if meta_keys is not None:
            if isinstance(meta_keys, str):
                meta_keys = (meta_keys, )
            else:
                assert isinstance(meta_keys, tuple), \
                    'meta_keys must be str or tuple.'
            self.meta_keys += meta_keys

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:
            - 'inputs' (dict[Tensor]): The forward data of models.
            - 'data_sample' (obj:`ReIDDataSample`): The meta info of the
                sample.
        """
        packed_results = dict(inputs=dict(), data_sample=None)
        assert 'img' in results, 'Missing the key ``img``.'
        _type = type(results['img'])
        label = results['gt_label']

        if _type == list:
            img = results['img']
            label = np.stack(label, axis=0)  # (N,)
            assert all([type(v) == _type for v in results.values()]), \
                'All items in the results must have the same type.'
        else:
            img = [results['img']]

        img = np.stack(img, axis=3)  # (H, W, C, N)
        img = img.transpose(3, 2, 0, 1)  # (N, C, H, W)
        img = np.ascontiguousarray(img)

        packed_results['inputs']['img'] = to_tensor(img)

        data_sample = ReIDDataSample()
        data_sample.set_gt_label(label)

        meta_info = dict()
        for key in self.meta_keys:
            if key == 'ori_shape' and 'ori_height' in results \
                    and 'ori_width' in results:
                if _type == list:
                    meta_info[key] = [(h, w) for h, w in zip(
                        results['ori_height'], results['ori_width'])]
                else:
                    meta_info[key] = (results['ori_height'],
                                      results['ori_width'])
            elif key in results:
                meta_info[key] = results[key]
        data_sample.set_metainfo(meta_info)
        packed_results['data_sample'] = data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str
