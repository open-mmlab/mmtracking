import os.path as osp

import numpy as np
from mmcv.utils import print_log
from mmdet.datasets import DATASETS

from mmtrack.core.evaluation import eval_sot_accuracy_robustness, eval_sot_eao
from .sot_test_dataset import SOTTestDataset


@DATASETS.register_module()
class VOTDataset(SOTTestDataset):
    """VOT dataset for the testing of single object tracking.

    The dataset doesn't support training mode.

    Note: The vot datasets using the mask annotation, such as VOT2020, is not
    supported now.
    """
    CLASSES = (0, )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_name = osp.basename(self.ann_file).rstrip('.json')
        # parameter, used for EAO evaluation, may vary by different vot
        # challenges.
        self.INTERVAL = dict(
            vot2018=[100, 356],
            vot2019=[46, 291],
            vot2020=[115, 755],
            vot2021=[115, 755])

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox annotations.

        Args:
            img_info (dict): image information.
            ann_info (list[dict]): Annotation information of an image. Each
                image only has one bbox annotation.
        Returns:
            dict: A dict containing the following keys: bboxes, labels.
            labels are not useful in SOT.
        """
        # The shape of gt_bboxes is (8, ), in [x1, y1, x2, y2, x3, y3, x4, y4]
        # format
        gt_bboxes = np.array(ann_info[0]['bbox'], dtype=np.float32)
        gt_labels = np.array(self.cat2label[ann_info[0]['category_id']])
        ann = dict(bboxes=gt_bboxes, labels=gt_labels)
        return ann

    # TODO support multirun test
    def evaluate(self, results, metric=['track'], logger=None, interval=None):
        """Evaluation in VOT protocol.

        Args:
            results (dict): Testing results of the dataset.
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

        eval_results = dict()
        if 'track' in metrics:
            assert len(self.data_infos) == len(results['track_bboxes'])
            print_log('Evaluate VOT Benchmark...', logger=logger)
            inds = []
            videos_wh = []
            ann_infos = []
            for i, info in enumerate(self.data_infos):
                if info['frame_id'] == 0:
                    inds.append(i)
                    videos_wh.append((info['width'], info['height']))

                ann_infos.append(self.get_ann_info(info))

            num_vids = len(inds)
            inds.append(len(self.data_infos))
            track_bboxes = []
            annotations = []
            for i in range(num_vids):
                bboxes_per_video = []
                for bbox in results['track_bboxes'][inds[i]:inds[i + 1]]:
                    # the last element of `bbox` is score.
                    if len(bbox) != 2:
                        # convert bbox format from (tl_x, tl_y, br_x, br_y) to
                        # (x1, y1, w, h)
                        bbox[2] -= bbox[0]
                        bbox[3] -= bbox[1]
                    bboxes_per_video.append(bbox[:-1])
                track_bboxes.append(bboxes_per_video)
                annotations.append(ann_infos[inds[i]:inds[i + 1]])

            interval = self.INTERVAL[self.dataset_name] if interval is None \
                else interval
            # anno_info is list[list[dict]]
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
