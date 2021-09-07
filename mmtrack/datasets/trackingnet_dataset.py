# Copyright (c) OpenMMLab. All rights reserved.
import json
import os
import os.path as osp
from collections import defaultdict

import numpy as np
from mmcv.utils import print_log
from mmdet.datasets import DATASETS

from mmtrack.core.evaluation import eval_sot_ope
from .coco_video_dataset import CocoVideoDataset


@DATASETS.register_module()
class TrackingNetTestDataset(CocoVideoDataset):
    """TrackingNet dataset for the testing of single object tracking.

    The dataset doesn't support training mode.
    """

    CLASSES = (0, )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # TODO del img_info; unify sot dataset class
    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox annotations.

        Args:
            img_info (dict): image information.
            ann_info (list[dict]): Annotation information of an image. Each
                image only has one bbox annotation.

        Returns:
            dict: A dict containing the following keys: bboxes, labels,
            ignore. labels are not useful in SOT.
        """
        gt_bboxes = np.array(ann_info[0]['bbox'], dtype=np.float32)
        # convert [x1, y1, w, h] to [x1, y1, x2, y2]
        gt_bboxes[2] += gt_bboxes[0]
        gt_bboxes[3] += gt_bboxes[1]
        gt_labels = np.array(self.cat2label[ann_info[0]['category_id']])
        ann = dict(bboxes=gt_bboxes, labels=gt_labels)
        return ann

    def format_results(self, results, resfile_path=None):
        """Format the results to txts (standard format for TrackingNet
        Challenge).

        Args:
            results (dict(list[ndarray])): Testing results of the dataset.
            resfile_path (str): Path to save the formatted results.
                Defaults to None.
        """
        # prepare saved dir
        assert resfile_path is not None, 'Please give key-value pair \
            like resfile_path=xxx in argparse'

        if not osp.isdir(resfile_path):
            os.makedirs(resfile_path, exist_ok=True)

        results = results['track_results']
        # transform results
        with open(self.ann_file, 'r') as f:
            info = json.load(f)
            video_info = info['videos']
            imgs_info = info['images']
        print('-------- Image Number: {} --------'.format(len(results)))

        new_results = defaultdict(list)
        for img_id, bbox in enumerate(results):
            img_info = imgs_info[img_id]
            assert img_info['id'] == img_id + 1, 'img id is not matched'
            video_name = img_info['file_name'].split('/')[0]
            new_results[video_name].append(results[img_id][:4])

        assert len(video_info) == len(
            new_results), 'video number is not right {}--{}'.format(
                len(video_info), len(new_results))

        # writing submitted results
        print('writing submitted results to {}'.format(resfile_path))
        for v_name, bboxes in new_results.items():
            vid_txt = osp.join(resfile_path, '{}.txt'.format(v_name))
            with open(vid_txt, 'w') as f:
                for i, bbox in enumerate(bboxes):
                    bbox = [
                        str(bbox[0]),
                        str(bbox[1]),
                        str(bbox[2] - bbox[0]),
                        str(bbox[3] - bbox[1])
                    ]
                    line = ','.join(bbox) + '\n'
                    f.writelines(line)

    def evaluate(self, results, metric=['track'], logger=None):
        """Evaluation in OPE protocol.

        Args:
            results (dict): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'track'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: OPE style evaluation metric (i.e. success,
            norm precision and precision).
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

        eval_results = dict()
        if 'track' in metrics:
            assert len(self.data_infos) == len(results['track_results'])
            print_log('Evaluate OPE Benchmark...', logger=logger)
            inds = [
                i for i, _ in enumerate(self.data_infos) if _['frame_id'] == 0
            ]
            num_vids = len(inds)
            inds.append(len(self.data_infos))

            track_bboxes = [
                list(
                    map(lambda x: x[:4],
                        results['track_results'][inds[i]:inds[i + 1]]))
                for i in range(num_vids)
            ]

            ann_infos = [self.get_ann_info(_) for _ in self.data_infos]
            ann_infos = [
                ann_infos[inds[i]:inds[i + 1]] for i in range(num_vids)
            ]
            track_eval_results = eval_sot_ope(
                results=track_bboxes, annotations=ann_infos)
            eval_results.update(track_eval_results)

            for k, v in eval_results.items():
                if isinstance(v, float):
                    eval_results[k] = float(f'{(v):.3f}')
            print_log(eval_results, logger=logger)

        return eval_results
