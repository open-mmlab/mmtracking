# Copyright (c) OpenMMLab. All rights reserved.
import random

import numpy as np
from mmdet.datasets import DATASETS

from .coco_video_dataset import CocoVideoDataset
from .parsers import CocoVID


@DATASETS.register_module()
class SOTTrainDataset(CocoVideoDataset):
    """Dataset for the training of single object tracking.

    The dataset doesn't support testing mode.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.load_as_video and not self.test_mode

    def load_video_anns(self, ann_file):
        """Load annotations from COCOVID style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation information from COCOVID api.
        """
        self.coco = CocoVID(ann_file, self.load_as_video)

        data_infos = []
        self.vid_ids = self.coco.get_vid_ids()
        for vid_id in self.vid_ids:
            info = self.coco.load_vids([vid_id])[0]
            data_infos.append(info)
        return data_infos

    def _filter_imgs(self):
        """Filter videos without ground truths."""
        valid_inds = []
        # obtain videos that contain annotation
        ids_with_ann = set(_['video_id'] for _ in self.coco.anns.values())

        valid_vid_ids = []
        for i, vid_info in enumerate(self.data_infos):
            vid_id = self.vid_ids[i]
            if self.filter_empty_gt and vid_id not in ids_with_ann:
                continue
            valid_inds.append(i)
            valid_vid_ids.append(vid_id)
        self.vid_ids = valid_vid_ids
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to video aspect ratio.

        It is not useful since all flags are set as 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def get_snippet_of_instance(self, idx):
        """Get a snippet of an instance in a video.

        Args:
            idx (int): Index of data.

        Returns:
            tuple: (snippet, image_id, instance_id), snippet is a list
            containing the successive image ids where the instance
            appears, image_id is a random sampled image id from the
            snippet.
        """
        vid_id = self.vid_ids[idx]
        instance_ids = self.coco.get_ins_ids_from_vid(vid_id)
        instance_id = np.random.choice(instance_ids)
        image_ids = self.coco.get_img_ids_from_ins_id(instance_id)
        if len(image_ids) > 1:
            snippets = np.split(
                image_ids,
                np.array(np.where(np.diff(image_ids) > 1)[0]) + 1)
            # remove isolated frame
            snippets = [s for s in snippets if len(s) > 1]
            # TODO: use random rather than -1
            snippet = snippets[-1].tolist()
        else:
            snippet = image_ids

        image_id = np.random.choice(snippet)
        return snippet, image_id, instance_id

    def ref_img_sampling(self,
                         snippet,
                         image_id,
                         instance_id,
                         frame_range=5,
                         pos_prob=0.8,
                         filter_key_img=False,
                         return_key_img=True,
                         **kwargs):
        """Get a search image for an instance in an exemplar image.

        If sampling a positive search image, the positive search image is
        randomly sampled from the exemplar image, where the sampled range is
        decided by `frame_range`.
        If sampling a negative search image, the negative search image and
        negative instance are randomly sampled from the entire dataset.

        Args:
            snippet (list[int]): The successive image ids where the instance
                appears.
            image_id (int): The id of exemplar image where the instance
                appears.
            instance_id (int): The id of the instance.
            frame_range (List(int) | int): The frame range of sampling a
                positive search image for the exemplar image. Default: 5.
            pos_prob (float): The probability of sampling a positive search
                image. Default: 0.8.
            filter_key_img (bool): If False, the exemplar image will be in the
                sampling candidates, otherwise, it is exclude. Default: False.
            return_key_img (bool): If True, the `image_id` and `instance_id`
                are returned, otherwise, not returned. Default: True.

        Returns:
            tuple: (image_ids, instance_ids, is_positive_pair), image_ids is
            a list that must contain search image id and may contain
            `image_id`, instance_ids is a list that must contain search
            instance id and may contain `instance_id`, is_positive_pair is
            a bool denoting positive or negative sample pair.
        """
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

        ref_image_ids = []
        ref_instance_ids = []
        if pos_prob > np.random.random():
            index = snippet.index(image_id)
            left = max(index + frame_range[0], 0)
            right = index + frame_range[1] + 1
            valid_ids = snippet[left:right]
            if filter_key_img and image_id in valid_ids:
                valid_ids.remove(image_id)
            ref_image_id = np.random.choice(valid_ids)
            ref_instance_id = instance_id
            is_positive_pair = True
        else:
            (ref_snippet, ref_image_id,
             ref_instance_id) = self.get_snippet_of_instance(
                 np.random.choice(range(len(self))))
            is_positive_pair = False

        ref_image_ids.append(ref_image_id)
        ref_instance_ids.append(ref_instance_id)

        if return_key_img:
            return [image_id, *ref_image_ids], \
                [instance_id, *ref_instance_ids], is_positive_pair
        else:
            return ref_image_ids, ref_instance_ids, is_positive_pair

    def prepare_results(self, img_id, instance_id, is_positive_pair):
        """Get training data and annotations.

        Args:
            img_id (int): The id of image.
            instance_id (int): The id of instance.
            is_positive_pair (bool): denoting positive or negative sample pair.

        Returns:
            dict: The information of training image and annotation.
        """
        img_info = self.coco.load_imgs([img_id])[0]
        img_info['filename'] = img_info['file_name']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_infos = self.coco.load_anns(ann_ids)
        ann = self._parse_ann_info(instance_id, ann_infos)

        result = dict(img_info=img_info, ann_info=ann)
        self.pre_pipeline(result)
        result['is_positive_pairs'] = is_positive_pair
        return result

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
            introduced by pipeline.
        """
        snippet, image_id, instance_id = self.get_snippet_of_instance(idx)
        image_ids, instance_ids, is_positive_pair = self.ref_img_sampling(
            snippet, image_id, instance_id, **self.ref_img_sampler)
        results = [
            self.prepare_results(img_id, instance_id, is_positive_pair)
            for img_id, instance_id in zip(image_ids, instance_ids)
        ]
        results = self.pipeline(results)
        return results

    def _parse_ann_info(self, instance_id, ann_infos):
        """Parse bbox annotation.

        Parse a given instance annotation from annotation infos of an image.

        Args:
            instance_id (int): The instance_id of an image need be parsed.
            ann_info (list[dict]): Annotation information of an image.

        Returns:
            dict: A dict containing the following keys: bboxes, labels. labels
            is set to `np.array([0])`.
        """
        has_instance_id = 0
        for ann_info in ann_infos:
            if ann_info['instance_id'] == instance_id:
                has_instance_id = 1
                break
        assert has_instance_id

        bbox = [[
            ann_info['bbox'][0], ann_info['bbox'][1],
            ann_info['bbox'][0] + ann_info['bbox'][2],
            ann_info['bbox'][1] + ann_info['bbox'][3]
        ]]
        ann = dict(
            bboxes=np.array(bbox, dtype=np.float32), labels=np.array([0]))
        return ann


@DATASETS.register_module()
class SOTQuotaTrainDataset(SOTTrainDataset):

    def __init__(self, visible_keys, max_gap, num_search_frames,
                 num_template_frames, *args, **kwargs):
        self.visible_keys = visible_keys
        self.max_gap = max_gap
        self.num_search_frames = num_search_frames
        self.num_template_frames = num_template_frames
        super().__init__(*args, **kwargs)
        self.is_video_dataset = 'videos' in self.coco.dataset

    def get_samples(self,
                    visible,
                    num_ids=1,
                    min_id=None,
                    max_id=None,
                    allow_invisible=False,
                    force_invisible=False):
        """ get num_ids frames between min_id and max_id in specific conditions
        args:
            visible - 1d Tensor indicating whether target is visible for each
                frame
            num_ids - number of frames to be samples
            min_id - Minimum allowed frame number
            max_id - Maximum allowed frame number
        returns:
            list - List of sampled frame numbers. None if not sufficient
                visible frames could be found.
        """
        if num_ids == 0:
            return []
        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(visible):
            max_id = len(visible)
        # get valid ids
        if force_invisible:
            valid_ids = [i for i in range(min_id, max_id) if not visible[i]]
        else:
            if allow_invisible:
                valid_ids = [i for i in range(min_id, max_id)]
            else:
                valid_ids = [i for i in range(min_id, max_id) if visible[i]]

        # No visible ids
        if len(valid_ids) == 0:
            return None

        return random.choices(valid_ids, k=num_ids)

    def get_visible_info(self):
        """Sample a sequence with enough visible frames.

        Returns:
            is_visible_ann (list): whether visible of each annotation about the
                instance in the sequence
            instance_id (int):
        """
        enough_visible_frames = False
        count = 0
        while not enough_visible_frames:
            count += 1
            if count > 100:
                raise Exception(
                    "--------Can't get enough visible instance------")
            # Sample a sequence
            vid_id = random.randint(
                1, len(self))  # The left and right of intervals are closed
            instance_ids = self.coco.get_ins_ids_from_vid(vid_id)
            instance_id = random.choice(instance_ids)
            img_ids = self.coco.get_img_ids_from_ins_id(instance_id)

            is_visible_ann = []
            new_ann_infos = []
            new_img_infos = []

            for img_id in img_ids:
                img_info = self.coco.load_imgs([img_id])[0]
                ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
                ann_infos = self.coco.load_anns(ann_ids)

                for ann in ann_infos:
                    if ann['instance_id'] == instance_id:
                        # coco dataset have different valid threshold
                        # (> 50 pixel).
                        # The visible information of trackingnet and coco only
                        # depend on the bbox size.
                        threshold_size = 0 if self.is_video_dataset else 50
                        valid = ann['bbox'][2] > threshold_size and ann[
                            'bbox'][3] > threshold_size
                        if not valid:
                            continue
                        visible = valid
                        if self.visible_keys is not None:
                            for key in self.visible_keys:
                                visible &= ~ann[key]
                        is_visible_ann.append(visible)

                        bbox = [[
                            ann['bbox'][0], ann['bbox'][1],
                            ann['bbox'][0] + ann['bbox'][2],
                            ann['bbox'][1] + ann['bbox'][3]
                        ]]
                        ann = dict(
                            bboxes=np.array(bbox, dtype=np.float32),
                            labels=np.array([0]))

                        new_ann_infos.append(ann)
                        img_info['filename'] = img_info['file_name']
                        new_img_infos.append(img_info)
            # TODO fix unittest failed
            enough_visible_frames = sum(is_visible_ann) > 2 * (
                self.num_search_frames +
                self.num_template_frames) and len(is_visible_ann) > 0
            # enough_visible_frames = sum(is_visible_ann) > (
            #     self.num_search_frames + self.num_template_frames)
            enough_visible_frames = enough_visible_frames or not \
                self.is_video_dataset

        return is_visible_ann, new_ann_infos, new_img_infos

    def sampling_trident(self, is_visible_ann):
        template_ann_ids_extra = []
        sampling_count = 0
        while None in template_ann_ids_extra or len(
                template_ann_ids_extra) == 0:
            template_ann_ids_extra = []
            # first randomly sample two frames from a video
            template_ann_id1 = self.get_samples(is_visible_ann, num_ids=1)
            search_ann_ids = self.get_samples(is_visible_ann, num_ids=1)
            # get the dynamic template id
            for max_gap in self.max_gap:
                if template_ann_id1[0] >= search_ann_ids[0]:
                    min_id, max_id = search_ann_ids[
                        0], search_ann_ids[0] + max_gap
                else:
                    min_id, max_id = search_ann_ids[
                        0] - max_gap, search_ann_ids[0]

                f_id = self.get_samples(
                    is_visible_ann,
                    num_ids=1,
                    min_id=min_id,
                    max_id=max_id,
                    allow_invisible=True)
                if f_id is None:
                    template_ann_ids_extra += [None]
                else:
                    template_ann_ids_extra += f_id
            sampling_count += 1
            if sampling_count > 100:
                print('-------Not sampling valid extra template. Stop'
                      'sampling and use the first template-------')
                template_ann_ids_extra = [template_ann_id1] * len(self.max_gap)

        all_ann_ids = template_ann_id1 + template_ann_ids_extra + \
            search_ann_ids
        return all_ann_ids

    def prepare_results(self, inds_intra_video, ann_infos, img_infos):
        """Get training data and annotations.

        Args:
            ann_id (int): The id of annotation.
            img_info (dict):

        Returns:
            dict: The information of training image and annotation.
        """
        assert len(inds_intra_video) == (
            self.num_template_frames + self.num_search_frames)
        results = []
        for i in range(self.num_template_frames):
            index = inds_intra_video[i]
            result = dict(img_info=img_infos[index], ann_info=ann_infos[index])
            self.pre_pipeline(result)
            results.append(result)

        for i in range(self.num_search_frames):
            index = inds_intra_video[self.num_template_frames + i]
            result = dict(img_info=img_infos[index], ann_info=ann_infos[index])
            self.pre_pipeline(result)
            results.append(result)
        return results

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data. Ignore it.

        Returns:
            dict: Training data and annotation after pipeline with new keys
            introduced by pipeline.
        """

        is_visible_ann, ann_infos, img_infos = self.get_visible_info()
        inds_intra_video = self.sampling_trident(is_visible_ann)
        results = self.prepare_results(inds_intra_video, ann_infos, img_infos)
        results = self.pipeline(results)
        return results
