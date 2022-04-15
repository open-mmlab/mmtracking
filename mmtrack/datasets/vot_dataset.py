# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import time

import mmcv
import numpy as np
from mmcv.utils import print_log
from mmdet.datasets import DATASETS

from mmtrack.core.evaluation import eval_sot_accuracy_robustness, eval_sot_eao
from .base_sot_dataset import BaseSOTDataset


@DATASETS.register_module()
class VOTDataset(BaseSOTDataset):
    """VOT dataset of single object tracking.

    The dataset is only used to test.
    """

    def __init__(self, dataset_type='vot2018', *args, **kwargs):
        """Initialization of SOT dataset class.

        Args:
            dataset_type (str, optional): The type of VOT challenge. The
                optional values are in ['vot2018', 'vot2018_lt',
                'vot2019', 'vot2019_lt', 'vot2020', 'vot2021']
        """
        assert dataset_type in [
            'vot2018', 'vot2018_lt', 'vot2019', 'vot2019_lt', 'vot2020',
            'vot2021'
        ]
        self.dataset_type = dataset_type
        super().__init__(*args, **kwargs)
        # parameter, used for EAO evaluation, may vary by different vot
        # challenges.
        self.INTERVAL = dict(
            vot2018=[100, 356],
            vot2019=[46, 291],
            vot2020=[115, 755],
            vot2021=[115, 755])

    def load_data_infos(self, split='test'):
        """Load dataset information.

        Args:
            split (str, optional): Dataset split. Defaults to 'test'.

        Returns:
            list[dict]: The length of the list is the number of videos. The
                inner dict is in the following format:
                    {
                        'video_path': the video path
                        'ann_path': the annotation path
                        'start_frame_id': the starting frame number contained
                            in the image name
                        'end_frame_id': the ending frame number contained in
                            the image name
                        'framename_template': the template of image name
                    }
        """
        print('Loading VOT dataset...')
        start_time = time.time()
        data_infos = []
        data_infos_str = self.loadtxt(
            self.ann_file, return_array=False).split('\n')
        # the first line of annotation file is a dataset comment.
        for line in data_infos_str[1:]:
            # compatible with different OS.
            line = line.strip().replace('/', os.sep).split(',')
            data_info = dict(
                video_path=line[0],
                ann_path=line[1],
                start_frame_id=int(line[2]),
                end_frame_id=int(line[3]),
                framename_template='%08d.jpg')
            data_infos.append(data_info)
        print(f'VOT dataset loaded! ({time.time()-start_time:.2f} s)')
        return data_infos

    def get_ann_infos_from_video(self, video_ind):
        """Get bboxes annotation about the instance in a video.

        Args:
            video_ind (int): video index

        Returns:
            ndarray: in [N, 8] shape. The N is the bbox number and the bbox
                is in (x1, y1, x2, y2, x3, y3, x4, y4) format.
        """
        bboxes = self.get_bboxes_from_video(video_ind)
        if bboxes.shape[1] == 4:
            x1, y1 = bboxes[:, 0], bboxes[:, 1],
            x2, y2 = bboxes[:, 0] + bboxes[:, 2], bboxes[:, 1],
            x3, y3 = bboxes[:, 0] + bboxes[:, 2], bboxes[:, 1] + bboxes[:, 3]
            x4, y4 = bboxes[:, 0], bboxes[:, 1] + bboxes[:, 3],
            bboxes = np.stack((x1, y1, x2, y2, x3, y3, x4, y4), axis=-1)

        visible_info = self.get_visibility_from_video(video_ind)
        # bboxes in VOT datasets are all valid
        bboxes_isvalid = np.array([True] * len(bboxes), dtype=np.bool_)
        ann_infos = dict(
            bboxes=bboxes, bboxes_isvalid=bboxes_isvalid, **visible_info)
        return ann_infos

    # TODO support multirun test
    def evaluate(self, results, metric=['track'], logger=None, interval=None):
        """Evaluation in VOT protocol.

        Args:
            results (dict): Testing results of the dataset. The tracking bboxes
                are in (tl_x, tl_y, br_x, br_y) format.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'track'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            interval (list): an specified interval in EAO curve used to
                calculate the EAO score. There are different settings in
                different VOT challenges.
        Returns:
            dict[str, float]:
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
        # annotations are in list[ndarray] format
        annotations = []
        for video_ind in range(len(self.data_infos)):
            bboxes = self.get_ann_infos_from_video(video_ind)['bboxes']
            annotations.append(bboxes)

        # tracking_bboxes converting code
        eval_results = dict()
        if 'track' in metrics:
            assert len(self) == len(
                results['track_bboxes']
            ), f"{len(self)} == {len(results['track_bboxes'])}"
            print_log('Evaluate VOT Benchmark...', logger=logger)
            track_bboxes = []
            start_ind = end_ind = 0
            videos_wh = []
            for data_info in self.data_infos:
                num = data_info['end_frame_id'] - data_info[
                    'start_frame_id'] + 1
                end_ind += num

                bboxes_per_video = []
                # results are in dict(track_bboxes=list[ndarray]) format
                # track_bboxes are in list[list[ndarray]] format
                for bbox in results['track_bboxes'][start_ind:end_ind]:
                    # the last element of `bbox` is score.
                    if len(bbox) != 2:
                        # convert bbox format from (tl_x, tl_y, br_x, br_y) to
                        # (x1, y1, w, h)
                        bbox[2] -= bbox[0]
                        bbox[3] -= bbox[1]

                    bboxes_per_video.append(bbox[:-1])

                track_bboxes.append(bboxes_per_video)
                start_ind += num

                # read one image in the video to get video width and height
                filename = osp.join(self.img_prefix, data_info['video_path'],
                                    data_info['framename_template'] % 1)
                img = mmcv.imread(
                    filename, file_client_args=self.file_client_args)
                videos_wh.append((img.shape[1], img.shape[0]))

            interval = self.INTERVAL[self.dataset_type] if interval is None \
                else interval

            eao_score = eval_sot_eao(
                results=track_bboxes,
                annotations=annotations,
                videos_wh=videos_wh,
                interval=interval)
            eval_results.update(eao_score)

            accuracy_robustness = eval_sot_accuracy_robustness(
                results=track_bboxes,
                annotations=annotations,
                videos_wh=videos_wh)
            eval_results.update(accuracy_robustness)
            for k, v in eval_results.items():
                if isinstance(v, float):
                    eval_results[k] = float(f'{(v):.4f}')
            print_log(eval_results, logger=logger)
        return eval_results
