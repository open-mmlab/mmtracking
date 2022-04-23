# Copyright (c) OpenMMLab. All rights reserved.
import random

import numpy as np
from mmcv.utils import print_log
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class TridentSampling(object):
    """Multitemplate-style sampling in a trident manner. It's firstly used in
    `STARK <https://arxiv.org/abs/2103.17154.>`_.

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
                 num_search_frames=1,
                 num_template_frames=2,
                 max_frame_range=[200],
                 cls_pos_prob=0.5,
                 train_cls_head=False,
                 min_num_frames=20):
        assert num_template_frames >= 2
        assert len(max_frame_range) == num_template_frames - 1
        self.num_search_frames = num_search_frames
        self.num_template_frames = num_template_frames
        self.max_frame_range = max_frame_range
        self.train_cls_head = train_cls_head
        self.cls_pos_prob = cls_pos_prob
        self.min_num_frames = min_num_frames

    def random_sample_inds(self,
                           video_visibility,
                           num_samples=1,
                           frame_range=None,
                           allow_invisible=False,
                           force_invisible=False):
        """Random sampling a specific number of samples from the specified
        frame range of the video. It also considers the visibility of each
        frame.

        Args:
            video_visibility (ndarray): the visibility of each frame in the
                video.
            num_samples (int, optional): the number of samples. Defaults to 1.
            frame_range (list | None, optional): the frame range of sampling.
                Defaults to None.
            allow_invisible (bool, optional): whether to allow to get invisible
                samples. Defaults to False.
            force_invisible (bool, optional): whether to force to get invisible
                samples. Defaults to False.

        Returns:
            List: The sampled frame indexes.
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

    def sampling_trident(self, video_visibility):
        """Sampling multiple template images and one search images in one
        video.

        Args:
            video_visibility (ndarray): the visibility of each frame in the
                video.

        Returns:
            List: the indexes of template and search images.
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

    def prepare_data(self, video_info, sampled_inds, with_label=False):
        """Prepare sampled training data according to the sampled index.

        Args:
            video_info (dict): the video information. It contains the keys:
                ['bboxes','bboxes_isvalid','filename','frame_ids',
                'video_id','visible'].
            sampled_inds (list[int]): the sampled frame indexes.
            with_label (bool, optional): whether to recode labels in ann infos.
                Only set True in classification training. Defaults to False.

        Returns:
            List[dict]: contains the information of sampled data.
        """
        extra_infos = {}
        for key, info in video_info.items():
            if key in [
                    'bbox_fields', 'mask_fields', 'seg_fields', 'img_prefix'
            ]:
                extra_infos[key] = info

        bboxes = video_info['bboxes']
        results = []
        for frame_ind in sampled_inds:
            if with_label:
                ann_info = dict(
                    bboxes=np.expand_dims(bboxes[frame_ind], axis=0),
                    labels=np.array([1.], dtype=np.float32))
            else:
                ann_info = dict(
                    bboxes=np.expand_dims(bboxes[frame_ind], axis=0))
            img_info = dict(
                filename=video_info['filename'][frame_ind],
                frame_id=video_info['frame_ids'][frame_ind],
                video_id=video_info['video_id'])
            result = dict(img_info=img_info, ann_info=ann_info, **extra_infos)
            results.append(result)
        return results

    def prepare_cls_data(self, video_info, video_info_another, sampled_inds):
        """Prepare the sampled classification training data according to the
        sampled index.

        Args:
            video_info (dict): the video information. It contains the keys:
                ['bboxes','bboxes_isvalid','filename','frame_ids',
                'video_id','visible'].
            video_info_another (dict): the another video information. It's only
                used to get negative samples in classification train. It
                contains the keys: ['bboxes','bboxes_isvalid','filename',
                'frame_ids','video_id','visible'].
            sampled_inds (list[int]): the sampled frame indexes.

        Returns:
            List[dict]: contains the information of sampled data.
        """
        results = self.prepare_data(
            video_info,
            sampled_inds[:self.num_template_frames],
            with_label=True)

        if random.random() < self.cls_pos_prob:
            pos_search_samples = self.prepare_data(
                video_info, sampled_inds[-self.num_search_frames:])
            for sample in pos_search_samples:
                sample['ann_info']['labels'] = np.array([1], dtype=np.float32)
            results.extend(pos_search_samples)
        else:
            if self.is_video_data:
                neg_search_ind = self.random_sample_inds(
                    video_info_another['bboxes_isvalid'], num_samples=1)
                # may not get valid negative sample in current video
                if neg_search_ind[0] is None:
                    return None
                neg_search_samples = self.prepare_data(video_info_another,
                                                       neg_search_ind)
            else:
                neg_search_samples = self.prepare_data(video_info_another, [0])

            for sample in neg_search_samples:
                sample['ann_info']['labels'] = np.array([0], dtype=np.float32)
            results.extend(neg_search_samples)
        return results

    def __call__(self, pair_video_infos):
        """
        Args:
            pair_video_infos (list[dict]): contains two video infos. Each video
                info contains the keys: ['bboxes','bboxes_isvalid','filename',
                'frame_ids','video_id','visible'].

        Returns:
            List[dict]: contains the information of sampled data.
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

        sampled_inds = np.array(self.sampling_trident(video_info['visible']))
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


@PIPELINES.register_module()
class PairSampling(object):
    """Pair-style sampling. It's used in `SiameseRPN++

    <https://arxiv.org/abs/1812.11703.>`_.

    Args:
        frame_range (List(int) | int): the sampling range of search
            frames in the same video for template frame. Defaults to 5.
        pos_prob (float, optional):  the probility of sampling positive
            sample pairs. Defaults to 0.8.
        filter_template_img (bool, optional): if False, the template image will
            be in the sampling search candidates, otherwise, it is exclude.
            Defaults to False.
    """

    def __init__(self, frame_range=5, pos_prob=0.8, filter_template_img=False):
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

    def prepare_data(self, video_info, sampled_inds, is_positive_pairs=False):
        """Prepare sampled training data according to the sampled index.

        Args:
            video_info (dict): the video information. It contains the keys:
                ['bboxes','bboxes_isvalid','filename','frame_ids',
                'video_id','visible'].
            sampled_inds (list[int]): the sampled frame indexes.
            is_positive_pairs (bool, optional): whether it's the positive
                pairs. Defaults to False.

        Returns:
            List[dict]: contains the information of sampled data.
        """
        extra_infos = {}
        for key, info in video_info.items():
            if key in [
                    'bbox_fields', 'mask_fields', 'seg_fields', 'img_prefix'
            ]:
                extra_infos[key] = info

        bboxes = video_info['bboxes']
        results = []
        for frame_ind in sampled_inds:
            ann_info = dict(bboxes=np.expand_dims(bboxes[frame_ind], axis=0))
            img_info = dict(
                filename=video_info['filename'][frame_ind],
                frame_id=video_info['frame_ids'][frame_ind],
                video_id=video_info['video_id'])
            result = dict(
                img_info=img_info,
                ann_info=ann_info,
                is_positive_pairs=is_positive_pairs,
                **extra_infos)
            results.append(result)
        return results

    def __call__(self, pair_video_infos):
        """
        Args:
            pair_video_infos (list[dict]): contains two video infos. Each video
                info contains the keys: ['bboxes','bboxes_isvalid','filename',
                'frame_ids','video_id','visible'].

        Returns:
            List[dict]: contains the information of sampled data.
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
                results.extend(
                    self.prepare_data(
                        video_info_another, [search_frame_ind],
                        is_positive_pairs=False))
        else:
            if self.pos_prob > np.random.random():
                results = self.prepare_data(
                    video_info, [0, 0], is_positive_pairs=True)
            else:
                results = self.prepare_data(
                    video_info, [0], is_positive_pairs=False)
                results.extend(
                    self.prepare_data(
                        video_info_another, [0], is_positive_pairs=False))
        return results


@PIPELINES.register_module()
class MatchInstances(object):
    """Matching objects on a pair of images.

    Args:
        skip_nomatch (bool, optional): Whether skip the pair of image
        during training when there are no matched objects. Default
        to True.
    """

    def __init__(self, skip_nomatch=True):
        self.skip_nomatch = skip_nomatch

    def _match_gts(self, instance_ids, ref_instance_ids):
        """Matching objects according to ground truth `instance_ids`.

        Args:
            instance_ids (ndarray): of shape (N1, ).
            ref_instance_ids (ndarray): of shape (N2, ).

        Returns:
            tuple: Matching results which contain the indices of the
            matched target.
        """
        ins_ids = list(instance_ids)
        ref_ins_ids = list(ref_instance_ids)
        match_indices = np.array([
            ref_ins_ids.index(i) if (i in ref_ins_ids and i > 0) else -1
            for i in ins_ids
        ])
        ref_match_indices = np.array([
            ins_ids.index(i) if (i in ins_ids and i > 0) else -1
            for i in ref_ins_ids
        ])
        return match_indices, ref_match_indices

    def __call__(self, results):
        if len(results) != 2:
            raise NotImplementedError('Only support match 2 images now.')

        match_indices, ref_match_indices = self._match_gts(
            results[0]['gt_instance_ids'], results[1]['gt_instance_ids'])
        nomatch = (match_indices == -1).all()
        if self.skip_nomatch and nomatch:
            return None
        else:
            results[0]['gt_match_indices'] = match_indices.copy()
            results[1]['gt_match_indices'] = ref_match_indices.copy()

        return results
