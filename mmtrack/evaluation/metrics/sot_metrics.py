# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import shutil
import tempfile
from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Optional, Sequence, Union

import mmengine
import numpy as np
from mmengine.logging import MMLogger
from mmengine.utils import mkdir_or_exist
from tabulate import tabulate

from mmtrack.registry import METRICS
from mmtrack.utils import format_video_level_show
from ..functional import (eval_sot_accuracy_robustness, eval_sot_eao,
                          eval_sot_ope)
from .base_video_metrics import BaseVideoMetric


@METRICS.register_module()
class SOTMetric(BaseVideoMetric):
    """SOT evaluation metrics.

    Args:
        metric (Union[str, Sequence[str]], optional): Metrics to be evaluated.
            Valid metrics are included in ``self.allowed_metrics``.
            Defaults to 'OPE'.
        metric_options (Optional[dict], optional): Options for calculating
            metrics. Defaults to dict(dataset_type='vot2018',
            only_eval_visible=False).
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
        saved_track_res_path (Optional[str], optional): The saved path of
            tracked results. Defaults to None.
        options_after_eval (Optional[dict], optional): The options after
            evaluation. Defaults to {}. The usage is the following:
            ```
                options_after_eval = dict(
                    saved_eval_res_file = './results/sot_res.json',
                    tracker_name = 'siamrpn++',
                    eval_show_video_indices = 10)
            ```
            Here, ``eval_show_video_indices`` is used to index a numpy.ndarray.
            It can be int (positive or negative) or list.
            ``saved_eval_res_file`` must be a json/yaml/pickle file.
    """
    default_prefix: Optional[str] = 'sot'
    allowed_metrics = ['OPE', 'VOT']
    allowed_metric_options = ['dataset_type', 'only_eval_visible']
    VOT_INTERVAL = dict(vot2018=[100, 356], vot2019=[46, 291])

    def __init__(self,
                 metric: Union[str, Sequence[str]] = 'OPE',
                 metric_options: Optional[dict] = dict(
                     dataset_type='vot2018', only_eval_visible=False),
                 format_only: bool = False,
                 outfile_prefix: Optional[str] = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 saved_track_res_path: Optional[str] = None,
                 options_after_eval: dict = {}) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.metrics = metric if isinstance(metric, list) else [metric]
        assert not (
            'OPE' in self.metrics and 'VOT' in self.metrics
        ), 'We can not evaluate one tracking result on both OPE and '
        'VOT metrics since the track result on VOT mode '
        'may be not the true bbox coordinates.'
        self.metric_options = metric_options
        for metric in self.metrics:
            if metric not in self.allowed_metrics:
                raise KeyError(
                    f'metric should be in {str(self.allowed_metrics)}, '
                    f'but got {metric}.')
        for metric_option in self.metric_options:
            if metric_option not in self.allowed_metric_options:
                raise KeyError(
                    f'metric option should be in {str(self.allowed_metric_options)}, '  # noqa: E501
                    f'but got {metric_option}.')
        self.outfile_prefix = outfile_prefix
        self.format_only = format_only
        self.saved_track_res_path = saved_track_res_path
        self.options_after_eval = options_after_eval
        self.preds_per_video, self.gts_per_video = [], []
        self.frame_ids, self.visible_per_video = [], []

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        for data_sample in data_samples:
            data_instance = data_sample['instances'][0]

            self.preds_per_video.append(
                data_sample['pred_track_instances']['bboxes'][0].cpu().numpy())
            if 'bbox' in data_instance:
                self.gts_per_video.append(data_instance['bbox'])
            else:
                assert self.format_only, 'If there is no ground truth '
                "bounding bbox, 'format_only' must be True"
            self.visible_per_video.append(data_instance['visible'])
            self.frame_ids.append(data_sample['frame_id'])

            if data_sample['frame_id'] == data_sample['video_length'] - 1:
                img_path_split = data_sample['img_path'].split(os.sep)
                # The ``img_path`` in LaSOT, OTB100 and VOT2018 have an extra
                # common directory outside the *.jpg file.
                video_name = img_path_split[-2] if img_path_split[-2] not in [
                    'img', 'color'
                ] else img_path_split[-3]
                result = dict(
                    video_name=video_name,
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
            results (List): The processed results of all data. The elements of
                the list are the processed results of one video.

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

        if self.saved_track_res_path is not None:
            self.save_tracked_results(all_pred_bboxes, all_video_names,
                                      self.saved_track_res_path)

        # 2. Fromat-only (Optional)
        if self.format_only:
            self.save_formatted_results(all_pred_bboxes, all_video_names)
            return dict()

        # 3. Evaluation (Optional)
        eval_results = OrderedDict()
        for metric in self.metrics:
            logger.info(f'Evaluating {metric}...')

            if metric == 'OPE':
                if self.metric_options.get('only_eval_visible', False):
                    results_ope = eval_sot_ope(all_pred_bboxes, all_gt_bboxes,
                                               all_visible)
                else:
                    results_ope = eval_sot_ope(all_pred_bboxes, all_gt_bboxes)

                ori_success = results_ope.pop('ori_success')
                ori_norm_precision = results_ope.pop('ori_norm_precision')
                ori_precision = results_ope.pop('ori_precision')
                eval_results.update(results_ope)

                saved_file_path = self.options_after_eval.get(
                    'saved_eval_res_file', None)
                if saved_file_path is not None:
                    if not saved_file_path.endswith(
                        ('.json', '.yml', '.yaml', '.pkl')):  # noqa: E125
                        raise TypeError(
                            f'Unsupported file format: {saved_file_path}. '
                            'Please specify a json, yaml or pickle file.')
                    if 'tracker_name' not in self.options_after_eval:
                        logger.warning(
                            'Not specify tracker name in the '
                            'argument options_after_eval and use the default '
                            "tracker name: 'anonymous_tracker'")
                        tracker_name = 'anonymous_tracker'
                    else:
                        tracker_name = self.options_after_eval['tracker_name']
                    mkdir_or_exist(osp.dirname(saved_file_path))
                    ori_eval_res = {
                        tracker_name:
                        dict(
                            success=np.mean(ori_success, axis=0),
                            norm_precision=np.mean(ori_norm_precision, axis=0),
                            precision=np.mean(ori_precision, axis=0))
                    }
                    mmengine.dump(ori_eval_res, saved_file_path)
                    logger.info(
                        'save evaluation results with different thresholds in '
                        f"'{saved_file_path}'")

                if self.options_after_eval.get('eval_show_video_indices',
                                               None) is not None:
                    success_per_video = np.mean(ori_success, axis=1)
                    norm_precision_per_video = ori_norm_precision[:, 20]
                    precision_per_video = ori_precision[:, 20]

                    eval_show_results = format_video_level_show(
                        all_video_names, [
                            success_per_video, norm_precision_per_video,
                            precision_per_video
                        ],
                        sort_by_first_metric=True,
                        show_indices=self.
                        options_after_eval['eval_show_video_indices'])

                    logger.info('\n' + tabulate(
                        eval_show_results,
                        headers=[
                            'video_name', 'success', 'norm_precision',
                            'precision'
                        ]))

            elif metric == 'VOT':
                if 'interval' in self.metric_options:
                    interval = self.metric_options['interval']
                else:
                    interval = self.VOT_INTERVAL.get(
                        self.metric_options['dataset_type'], None)
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

    def save_formatted_results_got10k(self, results: List[List[np.ndarray]],
                                      video_names: List[str],
                                      outfile_prefix: str):
        """Save the formatted results in TrackingNet dataset for evaluation on
        the test server.

        Args:
            results (List[List[np.ndarray]]): The formatted results.
            video_names (List[str]): The video names.
            outfile_prefix (str): The prefix of output files.
        """
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

    def save_formatted_results_trackingnet(self,
                                           results: List[List[np.ndarray]],
                                           video_names: List[str],
                                           outfile_prefix: str):
        """Save the formatted results in TrackingNet dataset for evaluation on
        the test server.

        Args:
            results (List[List[np.ndarray]]): The formatted results.
            video_names (List[str]): The video names.
            outfile_prefix (str): The prefix of output files.
        """
        for result, video_name in zip(results, video_names):
            video_txt = osp.join(outfile_prefix, f'{video_name}.txt')
            with open(video_txt, 'w') as f:
                for bbox in result:
                    bbox = [
                        str(f'{bbox[0]:.4f}'),
                        str(f'{bbox[1]:.4f}'),
                        str(f'{(bbox[2] - bbox[0]):.4f}'),
                        str(f'{(bbox[3] - bbox[1]):.4f}')
                    ]
                    line = ','.join(bbox) + '\n'
                    f.writelines(line)

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

        dataset_type = self.metric_options.get('dataset_type', 'got10k')
        if dataset_type == 'got10k':
            self.save_formatted_results_got10k(results, video_names,
                                               outfile_prefix)
        elif dataset_type == 'trackingnet':
            self.save_formatted_results_trackingnet(results, video_names,
                                                    outfile_prefix)
        shutil.make_archive(outfile_prefix, 'zip', outfile_prefix)
        shutil.rmtree(outfile_prefix)
        logger.info(
            f'-------- The formatted results are stored in {outfile_prefix}.zip --------'  # noqa: E501
        )

    def save_tracked_results(self, results: List[List[np.ndarray]],
                             video_names: List[str], saved_path: str):
        """Save the tracked results.

        Args:
            results (List[List[np.ndarray]]): The tracked results.
            video_names (List[str]): The video names.
            saved_path (str): The saved path of tracked results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        if not osp.isdir(saved_path):
            os.makedirs(saved_path, exist_ok=True)

        self.save_formatted_results_trackingnet(results, video_names,
                                                saved_path)
        logger.info(
            f'-------- The whole tracked results are stored in the fold {saved_path}--------'  # noqa: E501
        )
