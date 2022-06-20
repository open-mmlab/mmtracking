# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import shutil
import tempfile
from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
from mmengine.logging import MMLogger

from mmtrack.core import (eval_sot_accuracy_robustness, eval_sot_eao,
                          eval_sot_ope)
from mmtrack.metrics import BaseVideoMetric
from mmtrack.registry import METRICS


@METRICS.register_module()
class SOTMetric(BaseVideoMetric):
    """SOT evaluation metrics.

    Args:
        metric (Union[str, Sequence[str]], optional): Metrics to be evaluated.
            Valid metrics are included in ``self.allowed_metrics``.
            Defaults to 'OPE'.
        metric_options (Optional[dict], optional): Options for calculating
            metrics. Defaults to dict(vot_dataset_type='vot2018').
        format_only (bool, optional): If True, only formatting the results to
            the official format and not performing evaluation.
            Defaults to False.
        outfile_prefix (Optional[str], optional): The prefix of json files. It
            includes the file path and the prefix of filename,
            e.g., "a/b/prefix". If not specified, a temp file will be created.
            Defaults to None.
        collect_device (str, optional): Device name used for collecting results
            from different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (Optional[str], optional): The prefix that will be added in the
            metric names to disambiguate homonymous metrics of different
            evaluators. If prefix is not provided in the argument,
            self.default_prefix will be used instead. Defaults to None.
    """
    default_prefix: Optional[str] = 'sot'
    allowed_metrics = ['OPE', 'VOT']
    VOT_INTERVAL = dict(vot2018=[100, 356], vot2019=[46, 291])

    def __init__(self,
                 metric: Union[str, Sequence[str]] = 'OPE',
                 metric_options: Optional[dict] = dict(
                     vot_dataset_type='vot2018', only_eval_visible=False),
                 format_only: bool = False,
                 outfile_prefix: Optional[str] = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.metrics = metric if isinstance(metric, list) else [metric]
        assert not (
            'OPE' in self.metrics and 'VOT' in self.metrics
        ), 'We can not evaluate one tracking result on both OPE and '
        'VOT metrics since the track result on VOT mode '
        'may be not the true bbox coordinates.'
        self.metrics_options = metric_options
        for metric in self.metrics:
            if metric not in self.allowed_metrics:
                raise KeyError(
                    f'metric should be in {str(self.allowed_metrics)}, '
                    f'but got {metric}.')
        self.outfile_prefix = outfile_prefix
        self.format_only = format_only
        self.preds_per_video, self.gts_per_video = [], []
        self.frame_ids, self.visible_per_video = [], []

    def process(self, data_batch: Sequence[dict],
                predictions: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[dict]): A batch of data
                from the dataloader.
            predictions (Sequence[dict]): A batch of outputs from
                the model.
        """

        for data, pred in zip(data_batch, predictions):
            data_sample = data['data_sample']
            data_instance = data_sample['instances'][0]

            self.preds_per_video.append(
                pred['pred_track_instances']['bboxes'][0].cpu().numpy())
            if 'bbox' in data_instance:
                self.gts_per_video.append(data_instance['bbox'])
            else:
                assert self.format_only, 'If there is no ground truth '
                "bounding bbox, 'format_only' must be True"
            self.visible_per_video.append(data_instance['visible'])
            self.frame_ids.append(data_sample['frame_id'])

            if data_sample['frame_id'] == data_sample['video_length'] - 1:
                result = dict(
                    video_name=data_sample['img_path'].split(os.sep)[-2],
                    video_id=data_sample['video_id'],
                    video_size=(data_sample['ori_shape'][1],
                                data_sample['ori_shape'][0]),
                    frame_ids=deepcopy(self.frame_ids),
                    # Collect the annotations and predictions of this video.
                    # We don't convert the ``preds_per_video`` to
                    # ``np.ndarray`` since the track results in SOT may not the
                    # tracking box in EAO metrics.
                    pred_bboxes=deepcopy(self.preds_per_video),
                    gt_bboxes=np.array(self.gts_per_video, dtype=np.float32),
                    visible=np.array(self.visible_per_video, dtype=bool))

                self.frame_ids.clear()
                self.preds_per_video.clear()
                self.gts_per_video.clear()
                self.visible_per_video.clear()

                self.results.append(result)
                break

    def compute_metrics(self, results: List) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (List): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        # 1. Convert the format of reuslts for evaluation.
        all_pred_bboxes, all_gt_bboxes = [], []
        all_video_names, all_video_sizes, all_visible = [], [], []
        for result in results:
            all_video_names.append(result['video_name'])
            all_video_sizes.append(result['video_size'])
            all_pred_bboxes.append(result['pred_bboxes'])
            all_gt_bboxes.append(result['gt_bboxes'])
            all_visible.append(result['visible'])

        # 2. Fromat-only (Optional)
        if self.format_only:
            self.save_formatted_results(all_pred_bboxes, all_video_names)
            return dict()

        # 3. Evaluation (Optional)
        eval_results = OrderedDict()
        for metric in self.metrics:
            logger.info(f'Evaluating {metric}...')

            if metric == 'OPE':
                if self.metrics_options.get('only_eval_visible', False):
                    results_ope = eval_sot_ope(all_pred_bboxes, all_gt_bboxes,
                                               all_visible)
                else:
                    results_ope = eval_sot_ope(all_pred_bboxes, all_gt_bboxes)
                eval_results.update(results_ope)
            elif metric == 'VOT':
                if 'interval' in self.metrics_options:
                    interval = self.metrics_options['interval']
                else:
                    interval = self.VOT_INTERVAL.get(
                        self.metrics_options['vot_dataset_type'], None)
                eao_scores = eval_sot_eao(
                    all_pred_bboxes,
                    all_gt_bboxes,
                    videos_wh=all_video_sizes,
                    interval=interval)
                eval_results.update(eao_scores)
                accuracy_robustness = eval_sot_accuracy_robustness(
                    all_pred_bboxes, all_gt_bboxes, videos_wh=all_video_sizes)
                eval_results.update(accuracy_robustness)

            else:
                raise KeyError(
                    f"metric '{metric}' is not supported. Please use the "
                    f'metric in {str(self.allowed_metrics)}')

        return eval_results

    def save_formatted_results(self, results: List[List[np.ndarray]],
                               video_names: List[str]):
        """Save the formatted results for evaluation on the test server.

        Args:
            results (List[List[np.ndarray]]): The formatted results.
            video_names (List[str]): The video names.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        # prepare saved dir
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = self.outfile_prefix

        if not osp.isdir(outfile_prefix):
            os.makedirs(outfile_prefix, exist_ok=True)

        # save the results in the standard format for the test server.
        for result, video_name in zip(results, video_names):
            video_outfile_dir = osp.join(outfile_prefix, video_name)
            if not osp.isdir(video_outfile_dir):
                os.makedirs(video_outfile_dir, exist_ok=True)
            video_bbox_txt = osp.join(video_outfile_dir,
                                      '{}_001.txt'.format(video_name))
            video_time_txt = osp.join(video_outfile_dir,
                                      '{}_time.txt'.format(video_name))
            with open(video_bbox_txt,
                      'w') as f_bbox, open(video_time_txt, 'w') as f_time:

                for bbox in result:
                    bbox = [
                        str(f'{bbox[0]:.4f}'),
                        str(f'{bbox[1]:.4f}'),
                        str(f'{(bbox[2] - bbox[0]):.4f}'),
                        str(f'{(bbox[3] - bbox[1]):.4f}')
                    ]
                    line = ','.join(bbox) + '\n'
                    f_bbox.writelines(line)
                    # We don't record testing time, so we set a default
                    # time in order to test on the server.
                    f_time.writelines('0.0001\n')

        shutil.make_archive(outfile_prefix, 'zip', outfile_prefix)
        shutil.rmtree(outfile_prefix)
        logger.info(
            f'-------- The formatted results are stored in {outfile_prefix}.zip --------'  # noqa: E501
        )
