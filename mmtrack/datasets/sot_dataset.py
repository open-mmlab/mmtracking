import os.path as osp
import random
from abc import ABCMeta, abstractmethod

import numpy as np
from mmcv.utils import print_log
from mmdet.datasets.pipelines import Compose
from torch.utils.data import Dataset

from mmtrack.core.evaluation import eval_sot_ope
from mmtrack.datasets import DATASETS


@DATASETS.register_module()
class SOTDataset(Dataset, metaclass=ABCMeta):
    """Dataset of single object tracking.

    The dataset can both support training and testing mode.
    """

    CLASSES = None

    def __init__(self, img_prefix, pipeline, split, test_mode=False, **kwargs):
        self.img_prefix = img_prefix
        self.split = split
        self.pipeline = Compose(pipeline)
        self.test_mode = test_mode
        self.is_video_dataset = True
        self.bbox_min_size = 0
        ''' The self.data_info is a list, which the length is the
            number of videos. The default content is in the following format:
            [
                {
                    'video_path': the video path
                    'ann_path': the annotation path
                    'start_frame': the starting frame number contained in the
                                    image name
                    'end_frame': the ending frame number contained in the image
                                    name
                    'num_digit': the number of digits in the image name
                },
                ...
            ]
        '''
        self.data_infos = self.load_data_infos(split=self.split)
        self.num_frames_per_video = []
        for video_ind in range(len(self.data_infos)):
            self.num_frames_per_video.append(self.get_len_per_video(video_ind))

    def __getitem__(self, ind):
        if self.test_mode:
            assert isinstance(ind, tuple)
            return self.prepare_test_img(ind[0], ind[1])
        else:
            return self.prepare_train_img(ind)

    @abstractmethod
    def load_data_infos(self, split='train'):
        pass

    def get_img_names_from_video(self, video_ind):
        img_names = []
        start_frame, end_frame = self.data_infos[video_ind][
            'start_frame'], self.data_infos[video_ind]['end_frame']
        num_digit = self.data_infos[video_ind]['num_digit']
        for frame_id in range(start_frame, end_frame + 1):
            img_names.append(
                osp.join(
                    self.data_infos[video_ind]['video_path'],
                    '{frame_id:0{num_digit}}.jpg'.format(
                        frame_id=frame_id, num_digit=num_digit)))
        return img_names

    def get_bboxes_from_video(self, video_ind):
        bbox_path = osp.join(self.img_prefix,
                             self.data_infos[video_ind]['ann_path'])
        bboxes = np.loadtxt(bbox_path, dtype=float, delimiter=',')
        if len(bboxes.shape) == 1:
            bboxes = np.expand_dims(bboxes, axis=0)

        end_frame, start_frame = self.data_infos[video_ind][
            'end_frame'], self.data_infos[video_ind]['start_frame']

        if not self.test_mode:
            assert len(bboxes) == (
                end_frame - start_frame +
                1), f'{len(bboxes)}, {end_frame}, {start_frame}'
        return bboxes

    def get_len_per_video(self, video_ind):
        return self.data_infos[video_ind]['end_frame'] - self.data_infos[
            video_ind]['start_frame'] + 1

    def get_visibility_from_video(self, video_ind):
        visible = np.array([True] * self.get_len_per_video(video_ind))
        return dict(visible=visible)

    def get_masks_from_video(self, video_ind):
        pass

    def get_ann_infos_from_video(self, video_ind):
        bboxes = self.get_bboxes_from_video(video_ind)
        visible_info = self.get_visibility_from_video(video_ind)
        bboxes_isvalid = (bboxes[:, 2] > self.bbox_min_size) & (
            bboxes[:, 3] > self.bbox_min_size)
        visible_info['visible'] = visible_info['visible'] & bboxes_isvalid
        bboxes[:, 2:] += bboxes[:, :2]
        ann_infos = dict(
            bboxes=bboxes, bboxes_isvalid=bboxes_isvalid, **visible_info)
        return ann_infos

    def get_img_infos_from_video(self, video_ind):
        img_names = self.get_img_names_from_video(video_ind)
        frame_ids = np.arange(self.get_len_per_video(video_ind))
        img_infos = dict(
            filename=img_names, frame_ids=frame_ids, video_id=video_ind)
        return img_infos

    def prepare_test_img(self, video_ind, frame_ind):
        """Get the data of one frame.
        Args:
            video_ind (int): video index (>=0)
            frame_ind (int): frame index (>=0)

        Returns:
            dict:
        """
        ann_infos = self.get_ann_infos_from_video(video_ind)
        img_infos = self.get_img_infos_from_video(video_ind)
        img_info = dict(
            filename=img_infos['filename'][frame_ind], frame_id=frame_ind)
        ann_info = dict(
            bboxes=ann_infos['bboxes'][frame_ind],
            visible=ann_infos['visible'][frame_ind])

        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        results = self.pipeline(results)
        return results

    def prepare_train_img(self, video_ind):
        """Get the results sampled from some videos
        Args:
            video_ind (int): video index (>=0)

        Returns:
            dict:s
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
            if results is None:
                continue
            return results

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
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
            logger ([type], optional): defaults to None.
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
        annotations = []
        for video_ind in range(len(self.data_infos)):
            bboxes = self.get_ann_infos_from_video(video_ind)['bboxes']
            annotations.append(bboxes)

        # tracking_bboxes converting code
        eval_results = dict()
        if 'track' in metrics:
            assert len(self) == len(results['track_bboxes'])
            print_log('Evaluate OPE Benchmark...', logger=logger)
            track_bboxes = []
            start_ind = end_ind = 0
            for num in self.num_frames_per_video:
                end_ind += num
                track_bboxes.append(
                    list(
                        map(lambda x: x[:4],
                            results['track_bboxes'][start_ind:end_ind])))
                start_ind += num

            # evaluation
            track_eval_results = eval_sot_ope(
                results=track_bboxes, annotations=annotations)
            eval_results.update(track_eval_results)

            for k, v in eval_results.items():
                if isinstance(v, float):
                    eval_results[k] = float(f'{(v):.3f}')
            print_log(eval_results, logger=logger)
