# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import random
from abc import ABCMeta, abstractmethod
from io import StringIO

import mmcv
import numpy as np
from addict import Dict
from mmcv.utils import print_log
from mmdet.datasets.pipelines import Compose
from torch.utils.data import Dataset

from mmtrack.core.evaluation import eval_sot_ope
from mmtrack.datasets import DATASETS


@DATASETS.register_module()
class BaseSOTDataset(Dataset, metaclass=ABCMeta):
    """Dataset of single object tracking. The dataset can both support training
    and testing mode.

    Args:
        img_prefix (str): Prefix in the paths of image files.
        pipeline (list[dict]): Processing pipeline.
        split (str): Dataset split.
        ann_file (str, optional): The file contains data information. It will
            be loaded and parsed in the `self.load_data_infos` function.
        test_mode (bool, optional): Default to False.
        bbox_min_size (int, optional): Only bounding boxes whose sizes are
            larger than `bbox_min_size` can be regarded as valid. Default to 0.
        only_eval_visible (bool, optional): Whether to only evaluate frames
            where object are visible. Default to False.
        file_client_args (dict, optional): Arguments to instantiate a
                FileClient. Default: dict(backend='disk').
    """

    # Compatible with MOT and VID Dataset class. The 'CLASSES' attribute will
    # be called in tools/train.py.
    CLASSES = None

    def __init__(self,
                 img_prefix,
                 pipeline,
                 split,
                 ann_file=None,
                 test_mode=False,
                 bbox_min_size=0,
                 only_eval_visible=False,
                 file_client_args=dict(backend='disk'),
                 **kwargs):
        self.img_prefix = img_prefix
        self.split = split
        self.pipeline = Compose(pipeline)
        self.ann_file = ann_file
        self.test_mode = test_mode
        self.bbox_min_size = bbox_min_size
        self.only_eval_visible = only_eval_visible
        self.file_client_args = file_client_args
        self.file_client = mmcv.FileClient(**file_client_args)
        # 'self.load_as_video' must be set to True in order to using
        # distributed video sampler to load dataset when testing.
        self.load_as_video = True
        ''' The self.data_info is a list, which the length is the
            number of videos. The default content is in the following format:
            [
                {
                    'video_path': the video path
                    'ann_path': the annotation path
                    'start_frame_id': the starting frame ID number contained in
                                    the image name
                    'end_frame_id': the ending frame ID number contained in the
                                    image name
                    'framename_template': the template of image name
                },
                ...
            ]
        '''
        self.data_infos = self.load_data_infos(split=self.split)
        self.num_frames_per_video = [
            self.get_len_per_video(video_ind)
            for video_ind in range(len(self.data_infos))
        ]
        # used to record the video information at the beginning of the video
        # test. Thus, we can avoid reloading the files of video information
        # repeatedly in all frames of one video.
        self.test_memo = Dict()

    def __getitem__(self, ind):
        if self.test_mode:
            assert isinstance(ind, tuple)
            # the first element in the tuple is the video index and the second
            # element in the tuple is the frame index
            return self.prepare_test_data(ind[0], ind[1])
        else:
            return self.prepare_train_data(ind)

    @abstractmethod
    def load_data_infos(self, split='train'):
        pass

    def loadtxt(self,
                filepath,
                dtype=float,
                delimiter=None,
                skiprows=0,
                return_array=True):
        file_string = self.file_client.get_text(filepath)
        if return_array:
            return np.loadtxt(
                StringIO(file_string),
                dtype=dtype,
                delimiter=delimiter,
                skiprows=skiprows)
        else:
            return file_string.strip()

    def get_bboxes_from_video(self, video_ind):
        """Get bboxes annotation about the instance in a video.

        Args:
            video_ind (int): video index

        Returns:
            ndarray: in [N, 4] shape. The N is the number of bbox and the bbox
                is in (x, y, w, h) format.
        """
        bbox_path = osp.join(self.img_prefix,
                             self.data_infos[video_ind]['ann_path'])
        bboxes = self.loadtxt(bbox_path, dtype=float, delimiter=',')
        if len(bboxes.shape) == 1:
            bboxes = np.expand_dims(bboxes, axis=0)

        end_frame_id = self.data_infos[video_ind]['end_frame_id']
        start_frame_id = self.data_infos[video_ind]['start_frame_id']

        if not self.test_mode:
            assert len(bboxes) == (
                end_frame_id - start_frame_id + 1
            ), f'{len(bboxes)} is not equal to {end_frame_id}-{start_frame_id}+1'  # noqa
        return bboxes

    def get_len_per_video(self, video_ind):
        """Get the number of frames in a video."""
        return self.data_infos[video_ind]['end_frame_id'] - self.data_infos[
            video_ind]['start_frame_id'] + 1

    def get_visibility_from_video(self, video_ind):
        """Get the visible information of instance in a video."""
        visible = np.array([True] * self.get_len_per_video(video_ind))
        return dict(visible=visible)

    def get_masks_from_video(self, video_ind):
        pass

    def get_ann_infos_from_video(self, video_ind):
        """Get annotation information in a video.

        Args:
            video_ind (int): video index

        Returns:
            dict: {'bboxes': ndarray in (N, 4) shape, 'bboxes_isvalid':
                ndarray, 'visible':ndarray}. The annotation information in some
                datasets may contain 'visible_ratio'. The bbox is in
                (x1, y1, x2, y2) format.
        """
        bboxes = self.get_bboxes_from_video(video_ind)
        # The visible information in some datasets may contain
        # 'visible_ratio'.
        visible_info = self.get_visibility_from_video(video_ind)
        bboxes_isvalid = (bboxes[:, 2] > self.bbox_min_size) & (
            bboxes[:, 3] > self.bbox_min_size)
        visible_info['visible'] = visible_info['visible'] & bboxes_isvalid
        bboxes[:, 2:] += bboxes[:, :2]
        ann_infos = dict(
            bboxes=bboxes, bboxes_isvalid=bboxes_isvalid, **visible_info)
        return ann_infos

    def get_img_infos_from_video(self, video_ind):
        """Get image information in a video.

        Args:
            video_ind (int): video index

        Returns:
            dict: {'filename': list[str], 'frame_ids':ndarray, 'video_id':int}
        """
        img_names = []
        start_frame_id = self.data_infos[video_ind]['start_frame_id']
        end_frame_id = self.data_infos[video_ind]['end_frame_id']
        framename_template = self.data_infos[video_ind]['framename_template']
        for frame_id in range(start_frame_id, end_frame_id + 1):
            img_names.append(
                osp.join(self.data_infos[video_ind]['video_path'],
                         framename_template % frame_id))
        frame_ids = np.arange(self.get_len_per_video(video_ind))
        img_infos = dict(
            filename=img_names, frame_ids=frame_ids, video_id=video_ind)
        return img_infos

    def prepare_test_data(self, video_ind, frame_ind):
        """Get testing data of one frame. We parse one video, get one frame
        from it and pass the frame information to the pipeline.

        Args:
            video_ind (int): video index
            frame_ind (int): frame index

        Returns:
            dict: testing data of one frame.
        """
        if self.test_memo.get('video_ind', None) != video_ind:
            self.test_memo.video_ind = video_ind
            self.test_memo.ann_infos = self.get_ann_infos_from_video(video_ind)
            self.test_memo.img_infos = self.get_img_infos_from_video(video_ind)
        assert 'video_ind' in self.test_memo and 'ann_infos' in \
            self.test_memo and 'img_infos' in self.test_memo

        img_info = dict(
            filename=self.test_memo.img_infos['filename'][frame_ind],
            frame_id=frame_ind)
        ann_info = dict(
            bboxes=self.test_memo.ann_infos['bboxes'][frame_ind],
            visible=self.test_memo.ann_infos['visible'][frame_ind])

        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        results = self.pipeline(results)
        return results

    def prepare_train_data(self, video_ind):
        """Get training data sampled from some videos. We firstly sample two
        videos from the dataset and then parse the data information. The first
        operation in the training pipeline is frames sampling.

        Args:
            video_ind (int): video index

        Returns:
            dict: training data pairs, triplets or groups.
        """
        while True:
            video_inds = random.choices(list(range(len(self))), k=2)
            pair_video_infos = []
            for video_index in video_inds:
                ann_infos = self.get_ann_infos_from_video(video_index)
                img_infos = self.get_img_infos_from_video(video_index)
                video_infos = dict(**ann_infos, **img_infos)
                self.pre_pipeline(video_infos)
                pair_video_infos.append(video_infos)

            results = self.pipeline(pair_video_infos)
            if results is not None:
                return results

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline.

        The following keys in dict will be called in the subsequent pipeline.
        """
        results['img_prefix'] = self.img_prefix
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []

    def __len__(self):
        if self.test_mode:
            return sum(self.num_frames_per_video)
        else:
            return len(self.data_infos)

    def evaluate(self, results, metric=['track'], logger=None):
        """Default evaluation standard is OPE.

        Args:
            results (dict(list[ndarray])): tracking results. The ndarray is in
                (x1, y1, x2, y2, score) format.
            metric (list, optional): defaults to ['track'].
            logger (logging.Logger | str | None, optional): defaults to None.
        """

        if isinstance(metric, list):
            metrics = metric
        elif isinstance(metric, str):
            metrics = [metric]
        else:
            raise TypeError('metric must be a list or a str.')
        allowed_metrics = ['track']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported.')

        # get all test annotations
        gt_bboxes = []
        visible_infos = []
        for video_ind in range(len(self.data_infos)):
            video_anns = self.get_ann_infos_from_video(video_ind)
            gt_bboxes.append(video_anns['bboxes'])
            visible_infos.append(video_anns['visible'])

        # tracking_bboxes converting code
        eval_results = dict()
        if 'track' in metrics:
            assert len(self) == len(
                results['track_bboxes']
            ), f"{len(self)} == {len(results['track_bboxes'])}"
            print_log('Evaluate OPE Benchmark...', logger=logger)
            track_bboxes = []
            start_ind = end_ind = 0
            for num in self.num_frames_per_video:
                end_ind += num
                track_bboxes.append(
                    list(
                        map(lambda x: x[:-1],
                            results['track_bboxes'][start_ind:end_ind])))
                start_ind += num

            if not self.only_eval_visible:
                visible_infos = None
            # evaluation
            track_eval_results = eval_sot_ope(
                results=track_bboxes,
                annotations=gt_bboxes,
                visible_infos=visible_infos)
            eval_results.update(track_eval_results)

            for k, v in eval_results.items():
                if isinstance(v, float):
                    eval_results[k] = float(f'{(v):.3f}')
            print_log(eval_results, logger=logger)
        return eval_results
