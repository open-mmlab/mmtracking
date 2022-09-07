# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from mmcv.transforms import BaseTransform, to_tensor
from mmdet.structures.mask import BitmapMasks
from mmengine.structures import InstanceData

from mmtrack.registry import TRANSFORMS
from mmtrack.structures import ReIDDataSample, TrackDataSample


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
                                             'instances', 'num_left_ref_imgs',
                                             'frame_stride')):
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
            - 'data_samples' (obj:`TrackDataSample`): The annotation info of
                the samples.
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
                    new_img_metas[f'{self.ref_prefix}_{key}'] = ref_img_metas
                else:
                    new_img_metas[f'{self.ref_prefix}_{key}'] = ref_img_metas[
                        0]

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

        packed_results['data_samples'] = data_sample
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
                         'scale_factor')

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
            - 'data_samples' (obj:`ReIDDataSample`): The meta info of the
                sample.
        """
        packed_results = dict(inputs=dict(), data_samples=None)
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

        packed_results['inputs'] = to_tensor(img)

        data_sample = ReIDDataSample()
        data_sample.set_gt_label(label)

        meta_info = dict()
        for key in self.meta_keys:
            meta_info[key] = results[key]
        data_sample.set_metainfo(meta_info)
        packed_results['data_samples'] = data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str
