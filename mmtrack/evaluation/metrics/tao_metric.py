# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple, Union

import mmcv
import numpy as np
from mmengine.logging import MMLogger

from mmtrack.registry import METRICS
from .base_video_metrics import BaseVideoMetric

try:
    import tao
    from tao.toolkit.tao import Tao, TaoEval
except ImportError:
    tao = None


@METRICS.register_module()
class TAOMetric(BaseVideoMetric):
    """mAP evaluation metrics for the TAO task.

    Args:
        metric (str | list[str]): Metrics to be evaluated.
            Defaults to 'tao_track_ap'.
        metric_items (List[str], optional): Metric result names to be
            recorded in the evaluation result. Defaults to None.
        outfile_prefix (str | None): The prefix of json files. It includes
            the file path and the prefix of filename, e.g., "a/b/prefix".
            If not specified, a temp file will be created. Defaults to None.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonyms metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
        format_only (bool): If True, only formatting the results to the
            official format and not performing evaluation. Defaults to False.
    """

    default_prefix: Optional[str] = 'tao'

    def __init__(self,
                 metric: Union[str, List[str]] = 'tao_track_ap',
                 metric_items: Optional[Sequence[str]] = None,
                 outfile_prefix: Optional[str] = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 format_only: bool = False) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        # tao evaluation metrics
        self.metrics = metric if isinstance(metric, list) else [metric]
        self.format_only = format_only
        allowed_metrics = ['tao_track_ap']
        for metric in self.metrics:
            if metric not in allowed_metrics:
                raise KeyError(
                    f"metric should be 'tao_track_ap', but got {metric}.")

        self.metric_items = metric_items
        self.outfile_prefix = outfile_prefix
        self.per_video_res = []
        self.img_ids = []
        self.cat_ids = []
        self._tao_meta_info = defaultdict(list)

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
            result = dict()
            pred = pred['pred_track_instances']
            frame_id = data['data_sample']['frame_id']
            video_length = data['data_sample']['video_length']

            result['img_id'] = data['data_sample']['img_id']
            result['bboxes'] = pred['bboxes'].cpu().numpy()
            result['scores'] = pred['scores'].cpu().numpy()
            result['labels'] = pred['labels'].cpu().numpy()
            result['instances_id'] = pred['instances_id'].cpu().numpy()

            # parse gt
            gt = dict()
            gt['width'] = data['data_sample']['ori_shape'][1]
            gt['height'] = data['data_sample']['ori_shape'][0]
            keys = [
                'frame_id', 'frame_index', 'neg_category_ids',
                'not_exhaustive_category_ids', 'img_id', 'video_id',
                'video_length'
            ]
            for key in keys:
                if key not in data['data_sample']:
                    raise KeyError(
                        f'The key {key} is not found in track_data_sample,'
                        f' please pass it into the meta_keys'
                        f' of the PackTrackInputs')
                gt[key] = data['data_sample'][key]

            # When the ground truth exists, get annotation from `instances`.
            # In general, it contains `bbox`, `bbox_label` and `instance_id`.
            if 'instances' in data['data_sample']:
                gt['anns'] = data['data_sample']['instances']
            else:
                gt['anns'] = dict()
            self.per_video_res.append((result, gt))

            if frame_id == video_length - 1:
                preds, gts = zip(*self.per_video_res)
                # format the results
                gt_results = self._format_one_video_gts(gts)
                pred_results = self._format_one_video_preds(preds)
                self.per_video_res.clear()
                # add converted result to the results list
                self.results.append((pred_results, gt_results))

    def compute_metrics(self, results: List) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (List): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        # split gt and prediction list
        tmp_pred_results, tmp_gt_results = zip(*results)
        gt_results = self.format_gts(tmp_gt_results)
        pred_results = self.format_preds(tmp_pred_results)

        if self.format_only:
            self.save_pred_results(pred_results)
            return dict()

        eval_results = dict()

        if 'tao_track_ap' in self.metrics:
            if tao is None:
                raise ImportError(
                    'Please run'
                    ' pip install git+https://github.com/TAO-Dataset/tao.git '
                    'to manually install tao')

            logger.info('Evaluating tracking results...')
            tao_gt = Tao(gt_results)
            tao_eval = TaoEval(tao_gt, pred_results)
            tao_eval.params.img_ids = self.img_ids
            tao_eval.params.cat_ids = self.cat_ids
            tao_eval.params.iou_thrs = np.array([0.5, 0.75])
            tao_eval.run()

            tao_eval.print_results()
            tao_results = tao_eval.get_results()
            for k, v in tao_results.items():
                if isinstance(k, str) and k.startswith('AP'):
                    key = 'track_{}'.format(k)
                    val = float('{:.3f}'.format(float(v)))
                    eval_results[key] = val

        return eval_results

    def format_gts(self, gts: Tuple[List]) -> dict:
        """Gather all ground-truth from self.results."""
        categories = []
        for id, name in enumerate(self.dataset_meta['CLASSES']):
            categories.append(dict(id=id + 1, name=name))
            self.cat_ids.append(id + 1)
        for img_info in self._tao_meta_info['images']:
            self.img_ids.append(img_info['id'])
        gt_results = dict(
            info=dict(),
            images=self._tao_meta_info['images'],
            categories=categories,
            videos=self._tao_meta_info['videos'],
            annotations=[],
            tracks=self._tao_meta_info['tracks'])

        ann_id = 1
        for gt_result in gts:
            for ann in gt_result:
                ann['id'] = ann_id
                gt_results['annotations'].append(ann)
                ann_id += 1

        return gt_results

    def format_preds(self, preds: Tuple[List]) -> List:
        """Gather all predictions from self.results."""
        pred_results = []
        for pred_result in preds:
            pred_results.extend(pred_result)
        return pred_results

    def _format_one_video_preds(self, pred_dicts: Tuple[dict]) -> List:
        """Convert the annotation to the format of YouTube-VIS.

        This operation is to make it easier to use the official eval API.

        Args:
            pred_dicts (Tuple[dict]): Prediction of the dataset.

        Returns:
            List: The formatted predictions.
        """
        # Collate preds scatters (tuple of dict to dict of list)
        preds = defaultdict(list)
        for pred in pred_dicts:
            for key in pred.keys():
                preds[key].append(pred[key])

        vid_infos = self._tao_meta_info['videos']
        json_results = []
        video_id = vid_infos[-1]['id']

        for img_id, bboxes, scores, labels, ins_ids in zip(
                preds['img_id'], preds['bboxes'], preds['scores'],
                preds['labels'], preds['instances_id']):
            for bbox, score, label, ins_id in zip(bboxes, scores, labels,
                                                  ins_ids):
                data = dict(
                    image_id=img_id,
                    bbox=[
                        bbox[0],
                        bbox[1],
                        bbox[2] - bbox[0],
                        bbox[3] - bbox[1],
                    ],
                    score=score,
                    category_id=label + 1,
                    track_id=ins_id,
                    video_id=video_id)
                json_results.append(data)

        return json_results

    def _format_one_video_gts(self, gt_dicts: Tuple[dict]) -> List:
        """Convert the annotation to the format of YouTube-VIS.

        This operation is to make it easier to use the official eval API.

        Args:
            gt_dicts (Tuple[dict]): Ground truth of the dataset.

        Returns:
            list: The formatted gts.
        """
        video_infos = []
        image_infos = []
        track_infos = []
        annotations = []
        instance_flag = dict()  # flag the ins_id is used or not

        # get video infos
        for gt_dict in gt_dicts:
            frame_id = gt_dict['frame_id']
            video_id = gt_dict['video_id']
            img_id = gt_dict['img_id']
            image_info = dict(
                id=img_id,
                width=gt_dict['width'],
                height=gt_dict['height'],
                video_id=video_id,
                frame_id=frame_id,
                frame_index=gt_dict['frame_index'],
                neg_category_ids=gt_dict['neg_category_ids'],
                not_exhaustive_category_ids=gt_dict[
                    'not_exhaustive_category_ids'],
                file_name='')
            image_infos.append(image_info)
            if frame_id == 0:
                video_info = dict(
                    id=video_id,
                    width=gt_dict['width'],
                    height=gt_dict['height'],
                    neg_category_ids=gt_dict['neg_category_ids'],
                    not_exhaustive_category_ids=gt_dict[
                        'not_exhaustive_category_ids'],
                    file_name='')
                video_infos.append(video_info)

            for ann in gt_dict['anns']:
                label = ann['bbox_label']
                bbox = ann['bbox']
                instance_id = ann['instance_id']
                coco_bbox = [
                    bbox[0],
                    bbox[1],
                    bbox[2] - bbox[0],
                    bbox[3] - bbox[1],
                ]

                annotation = dict(
                    id=-1,  # need update when all results have been collected
                    video_id=video_id,
                    image_id=img_id,
                    frame_id=frame_id,
                    bbox=coco_bbox,
                    track_id=instance_id,
                    instance_id=instance_id,
                    iscrowd=ann.get('ignore_flag', 0),
                    category_id=label + 1,
                    area=coco_bbox[2] * coco_bbox[3])
                if not instance_flag.get(instance_id, False):
                    track_info = dict(
                        id=instance_id,
                        category_id=label + 1,
                        video_id=video_id)
                    track_infos.append(track_info)
                    instance_flag[instance_id] = True
                annotations.append(annotation)

        # update tao meta info
        self._tao_meta_info['images'].extend(image_infos)
        self._tao_meta_info['videos'].extend(video_infos)
        self._tao_meta_info['tracks'].extend(track_infos)

        return annotations

    def save_pred_results(self, pred_results: List) -> None:
        """Save the results to a zip file.

        Args:
            pred_results (list): Testing results of the
                dataset.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = self.outfile_prefix
        mmcv.dump(pred_results, f'{outfile_prefix}.json')

        logger.info(f'save the results to {outfile_prefix}.json')
