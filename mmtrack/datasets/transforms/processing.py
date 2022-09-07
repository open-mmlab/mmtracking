# Copyright (c) OpenMMLab. All rights reserved.
import random
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
from mmcv.transforms import BaseTransform
from mmengine.logging import print_log

from mmtrack.registry import TRANSFORMS


@TRANSFORMS.register_module()
class DiMPSampling(BaseTransform):
    """DiMP-style sampling. It's firstly used in `DiMP.

    <https://arxiv.org/abs/1904.07220>`_.

    Required Keys:

    - img_paths
    - frame_ids
    - video_id
    - video_length
    - bboxes
    - instance_id (optional)
    - mask (optional)

    - seg_map_path (optional)

    Added Keys:

    - instances

      - bbox (np.float32)
      - bbox_label (np.int32)
      - frame_id (np.int32)
      - ignore_flag (np.bool)
      - img_path (str)

    Args:
        num_search_frames (int, optional): the number of search frames
        num_template_frames (int, optional): the number of template frames
        max_frame_range (list[int], optional): the max frame range of sampling
            a positive search image for the template image. Its length is equal
            to the number of extra templates, i.e., `num_template_frames`-1.
            Default length is 1.
        min_num_frames (int, optional): the min number of frames to be sampled.
    """

    def __init__(self,
                 num_search_frames: int = 3,
                 num_template_frames: int = 3,
                 max_frame_range: int = 200,
                 min_num_frames: int = 20):
        self.num_search_frames = num_search_frames
        self.num_template_frames = num_template_frames
        self.max_frame_range = max_frame_range
        self.min_num_frames = min_num_frames

    def random_sample_inds(self,
                           video_visibility: np.ndarray,
                           num_samples: int = 1,
                           frame_range: Optional[List] = None,
                           allow_invisible: bool = False,
                           force_invisible: bool = False) -> List[int]:
        """Random sampling a specific number of samples from the specified
        frame range of the video. It also considers the visibility of each
        frame.

        Args:
            video_visibility (np.ndarray): the visibility of each frame in the
                video.
            num_samples (int, optional): the number of samples. Defaults to 1.
            frame_range (list | None, optional): the frame range of sampling.
                Defaults to None.
            allow_invisible (bool, optional): whether to allow to get invisible
                samples. Defaults to False.
            force_invisible (bool, optional): whether to force to get invisible
                samples. Defaults to False.

        Returns:
            List[int]: The sampled frame indexes.
        """
        assert num_samples > 0
        if frame_range is None:
            frame_range = [0, len(video_visibility)]
        else:
            assert isinstance(frame_range, list) and len(frame_range) == 2
            frame_range[0] = max(0, frame_range[0])
            frame_range[1] = min(len(video_visibility), frame_range[1])

        video_visibility = np.asarray(video_visibility)
        visibility_in_range = video_visibility[frame_range[0]:frame_range[1]]
        # get indexes of valid samples
        if force_invisible:
            valid_inds = np.where(~visibility_in_range)[0] + frame_range[0]
        else:
            valid_inds = np.arange(
                *frame_range) if allow_invisible else np.where(
                    visibility_in_range)[0] + frame_range[0]

        # No valid samples
        if len(valid_inds) == 0:
            return [None] * num_samples

        return random.choices(valid_inds, k=num_samples)

    def sampling_frames(self, video_visibility: np.ndarray) -> List:
        """Sampling multiple template images and one search images in one
        video.

        Args:
            video_visibility (np.ndarray): the visibility of each frame in the
                video.

        Returns:
            List: the indexes of template and search images.
        """
        search_frame_inds = [None]
        gap_increase = 0
        if self.is_video_data:
            while search_frame_inds[0] is None:
                # first randomly sample two frames from a video
                base_frame_ind = self.random_sample_inds(
                    video_visibility,
                    num_samples=1,
                    frame_range=[
                        self.num_template_frames - 1,
                        len(video_visibility) - self.num_search_frames
                    ])

                prev_frame_inds = self.random_sample_inds(
                    video_visibility,
                    num_samples=self.num_template_frames - 1,
                    frame_range=[
                        base_frame_ind[0] - self.max_frame_range -
                        gap_increase, base_frame_ind[0]
                    ])

                if prev_frame_inds[0] is None:
                    gap_increase += 5
                    continue

                temp_frame_inds = base_frame_ind + prev_frame_inds
                search_frame_inds = self.random_sample_inds(
                    video_visibility,
                    num_samples=self.num_search_frames,
                    frame_range=[
                        temp_frame_inds[0] + 1, temp_frame_inds[0] +
                        self.max_frame_range + gap_increase
                    ])

                gap_increase += 5

            sampled_inds = temp_frame_inds + search_frame_inds
        else:
            sampled_inds = [0] * (
                self.num_template_frames + self.num_search_frames)

        return sampled_inds

    def prepare_data(self,
                     video_info: dict,
                     sampled_inds: List[int],
                     is_positive_pairs: bool = True,
                     results: Optional[dict] = None) -> Dict[str, List]:
        """Prepare sampled training data according to the sampled index.

        Args:
            video_info (dict): the video information. It contains the keys:
                ['bboxes', 'bboxes_isvalid', 'img_paths', 'frame_ids',
                'video_id', 'visible', 'video_length].
            sampled_inds (list[int]): the sampled frame indexes.
            is_positive_pairs (bool, optional): whether it's the positive
                pairs. Defaults to True.
            results (dict[list], optional): The prepared results which need to
                be updated. Defaults to None.

        Returns:
            Dict[str, List]: contains the information of sampled data.
        """
        if results is None:
            results = defaultdict(list)
        assert isinstance(results, dict)
        for frame_ind in sampled_inds:
            results['img_path'].append(video_info['img_paths'][frame_ind])
            results['frame_id'].append(video_info['frame_ids'][frame_ind])
            results['video_id'].append(video_info['video_id'])
            results['video_length'].append(video_info['video_length'])
            instance = [
                dict(
                    bbox=video_info['bboxes'][frame_ind],
                    bbox_label=np.array(is_positive_pairs, dtype=np.int32))
            ]
            results['instances'].append(instance)
        return results

    def transform(self,
                  pair_video_infos: List[dict]) -> Optional[Dict[str, List]]:
        """
        Args:
            pair_video_infos (list[dict]): contains two video infos. Each video
                info contains the keys: ['bboxes','bboxes_isvalid','filename',
                'frame_ids','video_id','visible'].

        Returns:
            Optional[Dict[str, List]]: contains the information of sampled
                data.
        """
        video_info, video_info_another = pair_video_infos
        self.is_video_data = len(video_info['frame_ids']) > 1 and len(
            video_info_another['frame_ids']) > 1
        enough_visible_frames = sum(video_info['visible']) > 2 * (
            self.num_search_frames + self.num_template_frames) and len(
                video_info['visible']) >= self.min_num_frames
        enough_visible_frames = enough_visible_frames or not \
            self.is_video_data

        if not enough_visible_frames:
            return None

        sampled_inds = np.array(self.sampling_frames(video_info['visible']))
        # the sizes of some bboxes may be zero, because extral templates may
        # get invalid bboxes.
        if not video_info['bboxes_isvalid'][sampled_inds].all():
            return None

        results = self.prepare_data(video_info, sampled_inds)

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'num_search_frames={self.num_search_frames}, '
        repr_str += f'num_template_frames={self.num_template_frames}, '
        repr_str += f'max_frame_range={self.max_frame_range})'
        repr_str += f'min_num_frames={self.min_num_frames})'
        return repr_str


@TRANSFORMS.register_module()
class TridentSampling(DiMPSampling):
    """Multitemplate-style sampling in a trident manner. It's firstly used in
    `STARK <https://arxiv.org/abs/2103.17154.>`_.

    The input in this transform is a list of dict. In each dict:

    Required Keys:

    - img_paths
    - frame_ids
    - video_id
    - video_length
    - bboxes
    - instance_id (optional)
    - mask (optional)

    - seg_map_path (optional)

    Added Keys:

    - instances

      - bbox (np.float32)
      - bbox_label (np.int32)
      - frame_id (np.int32)
      - ignore_flag (np.bool)
      - img_path (str)

    Args:
        num_search_frames (int, optional): the number of search frames
        num_template_frames (int, optional): the number of template frames
        max_frame_range (list[int], optional): the max frame range of sampling
            a positive search image for the template image. Its length is equal
            to the number of extra templates, i.e., `num_template_frames`-1.
            Default length is 1.
        cls_pos_prob (float, optional): the probility of sampling positive
            samples in classification training.
        train_cls_head (bool, optional): whether to train classification head.
        min_num_frames (int, optional): the min number of frames to be sampled.
    """

    def __init__(self,
                 num_search_frames: int = 1,
                 num_template_frames: int = 2,
                 max_frame_range: List[int] = [200],
                 min_num_frames: int = 20,
                 cls_pos_prob: float = 0.5,
                 train_cls_head: bool = False):
        assert num_template_frames >= 2
        assert len(max_frame_range) == num_template_frames - 1
        super().__init__(num_search_frames, num_template_frames,
                         max_frame_range, min_num_frames)
        self.train_cls_head = train_cls_head
        self.cls_pos_prob = cls_pos_prob

    def sampling_frames(self, video_visibility: np.ndarray) -> List[int]:
        """Sampling multiple template images and one search images in one
        video.

        Args:
            video_visibility (np.ndarray): the visibility of each frame in the
                video.

        Returns:
            List[int]: the indexes of template and search images.
        """
        extra_template_inds = [None]
        sampling_count = 0
        if self.is_video_data:
            while None in extra_template_inds:
                # first randomly sample two frames from a video
                template_ind, search_ind = self.random_sample_inds(
                    video_visibility, num_samples=2)

                # then sample the extra templates
                extra_template_inds = []
                for max_frame_range in self.max_frame_range:
                    # make the sampling range is near the template_ind
                    if template_ind >= search_ind:
                        min_ind, max_ind = search_ind, \
                            search_ind + max_frame_range
                    else:
                        min_ind, max_ind = search_ind - max_frame_range, \
                            search_ind
                    extra_template_index = self.random_sample_inds(
                        video_visibility,
                        num_samples=1,
                        frame_range=[min_ind, max_ind],
                        allow_invisible=False)[0]

                    extra_template_inds.append(extra_template_index)

                sampling_count += 1
                if sampling_count > 100:
                    print_log('-------Not sampling extra valid templates'
                              'successfully. Stop sampling and copy the'
                              'first template as extra templates-------')
                    extra_template_inds = [template_ind] * len(
                        self.max_frame_range)

            sampled_inds = [template_ind] + extra_template_inds + [search_ind]
        else:
            sampled_inds = [0] * (
                self.num_template_frames + self.num_search_frames)

        return sampled_inds

    def prepare_cls_data(self, video_info: dict, video_info_another: dict,
                         sampled_inds: List[int]) -> Dict[str, List]:
        """Prepare the sampled classification training data according to the
        sampled index.

        Args:
            video_info (dict): the video information. It contains the keys:
                ['bboxes', 'bboxes_isvalid', 'filename', 'frame_ids',
                'video_id', 'visible', 'video_length].
            video_info_another (dict): the another video information. It's only
                used to get negative samples in classification train. It
                contains the keys: ['bboxes','bboxes_isvalid','filename',
                'frame_ids','video_id','visible','video_length]].
            sampled_inds (list[int]): the sampled frame indexes.

        Returns:
            Dict[str, List]: contains the information of sampled data.
        """
        if random.random() < self.cls_pos_prob:
            results = self.prepare_data(
                video_info, sampled_inds, is_positive_pairs=True)
        else:
            results = self.prepare_data(
                video_info,
                sampled_inds[:self.num_template_frames],
                is_positive_pairs=False)

            if self.is_video_data:
                neg_search_ind = self.random_sample_inds(
                    video_info_another['bboxes_isvalid'], num_samples=1)
                # may not get valid negative sample in current video
                if neg_search_ind[0] is None:
                    return None
            else:
                neg_search_ind = [0]

            results = self.prepare_data(
                video_info_another,
                neg_search_ind,
                is_positive_pairs=False,
                results=results)

        return results

    def transform(self,
                  pair_video_infos: List[dict]) -> Optional[Dict[str, List]]:
        """
        Args:
            pair_video_infos (list[dict]): contains two video infos. Each video
                info contains the keys: ['bboxes','bboxes_isvalid','filename',
                'frame_ids','video_id','visible','video_length'].

        Returns:
            Optional[Dict[str, List]]: contains the information of sampled
                data. If not enough visible frames, return None.
        """
        video_info, video_info_another = pair_video_infos
        self.is_video_data = len(video_info['frame_ids']) > 1 and len(
            video_info_another['frame_ids']) > 1
        enough_visible_frames = sum(video_info['visible']) > 2 * (
            self.num_search_frames + self.num_template_frames) and len(
                video_info['visible']) >= self.min_num_frames
        enough_visible_frames = enough_visible_frames or not \
            self.is_video_data

        if not enough_visible_frames:
            return None

        sampled_inds = np.array(self.sampling_frames(video_info['visible']))
        # the sizes of some bboxes may be zero, because extral templates may
        # get invalid bboxes.
        if not video_info['bboxes_isvalid'][sampled_inds].all():
            return None

        if not self.train_cls_head:
            results = self.prepare_data(video_info, sampled_inds)
        else:
            results = self.prepare_cls_data(video_info, video_info_another,
                                            sampled_inds)

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'num_search_frames={self.num_search_frames}, '
        repr_str += f'num_template_frames={self.num_template_frames}, '
        repr_str += f'max_frame_range={self.max_frame_range}, '
        repr_str += f'cls_pos_prob={self.cls_pos_prob}, '
        repr_str += f'train_cls_head={self.train_cls_head}, '
        repr_str += f'min_num_frames={self.min_num_frames})'
        return repr_str


@TRANSFORMS.register_module()
class PairSampling(BaseTransform):
    """Pair-style sampling. It's used in `SiameseRPN++

    <https://arxiv.org/abs/1812.11703.>`_.

    The input in this transform is a list of dict. In each dict:

    Required Keys:

    - img_paths
    - frame_ids
    - video_id
    - video_length
    - bboxes
    - instance_id (optional)
    - mask (optional)

    - seg_map_path (optional)

    Added Keys:

    - instances

      - bbox (np.float32)
      - bbox_label (np.int32)
      - frame_id (np.int32)
      - ignore_flag (np.bool)
      - img_path (str)

    Args:
        frame_range (List(int) | int): The sampling range of search
            frames in the same video for template frame. Defaults to 5.
        pos_prob (float, optional):  The probility of sampling positive
            sample pairs. Defaults to 0.8.
        filter_template_img (bool, optional): If False, the template image will
            be in the sampling search candidates, otherwise, it is exclude.
            Defaults to False.
    """

    def __init__(self,
                 frame_range: int = 5,
                 pos_prob: float = 0.8,
                 filter_template_img: bool = False):
        assert pos_prob >= 0.0 and pos_prob <= 1.0
        if isinstance(frame_range, int):
            assert frame_range >= 0, 'frame_range can not be a negative value.'
            frame_range = [-frame_range, frame_range]
        elif isinstance(frame_range, list):
            assert len(frame_range) == 2, 'The length must be 2.'
            assert frame_range[0] <= 0 and frame_range[1] >= 0
            for i in frame_range:
                assert isinstance(i, int), 'Each element must be int.'
        else:
            raise TypeError('The type of frame_range must be int or list.')
        self.frame_range = frame_range
        self.pos_prob = pos_prob
        self.filter_template_img = filter_template_img

    def prepare_data(self,
                     video_info: dict,
                     sampled_inds: List[int],
                     is_positive_pairs: bool = False,
                     results: Optional[dict] = None) -> Dict[str, List]:
        """Prepare sampled training data according to the sampled index.

        Args:
            video_info (dict): the video information. It contains the keys:
                ['bboxes', 'bboxes_isvalid', 'img_paths', 'frame_ids',
                'video_id', 'visible', 'video_length].
            sampled_inds (list[int]): the sampled frame indexes.
            is_positive_pairs (bool, optional): whether it's the positive
                pairs. Defaults to False.
            results (dict[list], optional): The prepared results which need to
                be updated. Defaults to None.

        Returns:
            Dict[str, List]: contains the information of sampled data.
        """
        if results is None:
            results = defaultdict(list)
        assert isinstance(results, dict)
        for frame_ind in sampled_inds:
            results['img_path'].append(video_info['img_paths'][frame_ind])
            results['frame_id'].append(video_info['frame_ids'][frame_ind])
            results['video_id'].append(video_info['video_id'])
            results['video_length'].append(video_info['video_length'])
            instance = [
                dict(
                    bbox=video_info['bboxes'][frame_ind],
                    bbox_label=np.array(is_positive_pairs, dtype=np.int32))
            ]
            results['instances'].append(instance)
        return results

    def transform(self, pair_video_infos: List[dict]) -> dict:
        """
        Args:
            pair_video_infos (list[dict]): Containing the information of two
                videos. Each video information contains the keys:
                ['bboxes','bboxes_isvalid', 'img_paths', 'frame_ids',
                'video_id', 'visible', 'video_length'].

        Returns:
            dict: contains the information of sampled data.
        """
        video_info, video_info_another = pair_video_infos
        if len(video_info['frame_ids']) > 1 and len(
                video_info_another['frame_ids']) > 1:
            template_frame_ind = np.random.choice(len(video_info['frame_ids']))
            if self.pos_prob > np.random.random():
                left_ind = max(template_frame_ind + self.frame_range[0], 0)
                right_ind = min(template_frame_ind + self.frame_range[1],
                                len(video_info['frame_ids']))
                if self.filter_template_img:
                    ref_frames_inds = list(
                        range(left_ind, template_frame_ind)) + list(
                            range(template_frame_ind + 1, right_ind))
                else:
                    ref_frames_inds = list(range(left_ind, right_ind))
                search_frame_ind = np.random.choice(ref_frames_inds)
                results = self.prepare_data(
                    video_info, [template_frame_ind, search_frame_ind],
                    is_positive_pairs=True)
            else:
                search_frame_ind = np.random.choice(
                    len(video_info_another['frame_ids']))
                results = self.prepare_data(
                    video_info, [template_frame_ind], is_positive_pairs=False)
                results = self.prepare_data(
                    video_info_another, [search_frame_ind],
                    is_positive_pairs=False,
                    results=results)

        else:
            if self.pos_prob > np.random.random():
                results = self.prepare_data(
                    video_info, [0, 0], is_positive_pairs=True)
            else:
                results = self.prepare_data(
                    video_info, [0], is_positive_pairs=False)
                results = self.prepare_data(
                    video_info_another, [0],
                    is_positive_pairs=False,
                    results=results)

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'frame_range={self.frame_range}, '
        repr_str += f'pos_prob={self.pos_prob}, '
        repr_str += f'filter_template_img={self.filter_template_img})'
        return repr_str
