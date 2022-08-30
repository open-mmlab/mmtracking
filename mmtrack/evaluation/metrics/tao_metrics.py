# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple, Union

import mmengine
import numpy as np
from mmengine.logging import MMLogger

from mmtrack.registry import METRICS
from .base_video_metrics import BaseVideoMetric

try:
    import tao
    from tao.toolkit.tao import Tao, TaoEval
except ImportError:
    tao = None

try:
    import lvis
    from lvis import LVIS, LVISEval, LVISResults
except ImportError:
    lvis = None


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
        allowed_metrics = ['tao_track_ap', 'bbox']
        for metric in self.metrics:
            if metric not in allowed_metrics:
                raise KeyError(f"metric should be 'tao_track_ap' or 'bbox',"
                               f' but got {metric}.')

        self.metric_items = metric_items
        self.outfile_prefix = outfile_prefix
        self.per_video_res = []
        self.img_ids = []

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
            result = dict()
            pred_track = data_sample['pred_track_instances']
            pred_det = data_sample['pred_det_instances']
            frame_id = data_sample['frame_id']
            video_length = data_sample['video_length']

            result['img_id'] = data_sample['img_id']
            result['track_bboxes'] = pred_track['bboxes'].cpu().numpy()
            result['track_scores'] = pred_track['scores'].cpu().numpy()
            result['track_labels'] = pred_track['labels'].cpu().numpy()
            result['track_instances_id'] = pred_track['instances_id'].cpu(
            ).numpy()

            result['det_bboxes'] = pred_det['bboxes'].cpu().numpy()
            result['det_scores'] = pred_det['scores'].cpu().numpy()
            result['det_labels'] = pred_det['labels'].cpu().numpy()

            # parse gt
            gt = dict()
            gt['width'] = data_sample['ori_shape'][1]
            gt['height'] = data_sample['ori_shape'][0]
            keys = [
                'frame_id', 'frame_index', 'neg_category_ids',
                'not_exhaustive_category_ids', 'img_id', 'video_id',
                'video_length'
            ]
            for key in keys:
                if key not in data_sample:
                    raise KeyError(
                        f'The key {key} is not found in track_data_sample,'
                        f' please pass it into the meta_keys'
                        f' of the PackTrackInputs')
                gt[key] = data_sample[key]

            # When the ground truth exists, get annotation from `instances`.
            # In general, it contains `bbox`, `bbox_label` and `instance_id`.
            if 'instances' in data_sample:
                gt['anns'] = data_sample['instances']
            else:
                gt['anns'] = dict()
            self.per_video_res.append((result, gt))

            if frame_id == video_length - 1:
                preds, gts = zip(*self.per_video_res)
                # format the results
                gt_results, tao_meta_info = self._format_one_video_gts(gts)
                pred_track_results, pred_det_results = \
                    self._format_one_video_preds(preds, tao_meta_info)
                self.per_video_res.clear()
                # add converted result to the results list
                self.results.append((pred_track_results, pred_det_results,
                                     gt_results, tao_meta_info))

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
        tmp_pred_track_results, tmp_pred_det_results, \
            tmp_gt_results, tmp_meta_info = zip(*results)
        tao_meta_info = self.format_meta(tmp_meta_info)
        gt_results = self.format_gts(tmp_gt_results, tao_meta_info)
        pred_track_results = self.format_preds(tmp_pred_track_results)
        pred_det_results = self.format_preds(tmp_pred_det_results)

        if 'bbox' in self.metrics:
            # LVIS api only supports reading from files, hence,
            # save the json result to tmp dir
            tmp_dir = tempfile.TemporaryDirectory()
            pred_det_results_path = f'{tmp_dir.name}/tao_bbox.json'
            gt_results_path = f'{tmp_dir.name}/tao_gt.json'
            mmengine.dump(pred_det_results, pred_det_results_path)
            mmengine.dump(gt_results, gt_results_path)

        if self.format_only:
            self.save_pred_results(pred_track_results, 'track')
            self.save_pred_results(pred_det_results, 'det')
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
            tao_eval = TaoEval(tao_gt, pred_track_results)
            tao_eval.params.img_ids = self.img_ids
            tao_eval.params.cat_ids = list(
                self.dataset_meta['categories'].keys())
            tao_eval.params.iou_thrs = np.array([0.5, 0.75])
            tao_eval.run()

            tao_eval.print_results()
            tao_results = tao_eval.get_results()
            for k, v in tao_results.items():
                if isinstance(k, str) and k.startswith('AP'):
                    key = 'track_{}'.format(k)
                    val = float('{:.3f}'.format(float(v)))
                    eval_results[key] = val

        if 'bbox' in self.metrics:
            if lvis is None:
                raise ImportError(
                    'Please run'
                    ' pip install git+https://github.com/lvis-dataset/lvis-api.git '  # noqa
                    'to manually install lvis')

            logger.info('Evaluating detection results...')
            lvis_gt = LVIS(gt_results_path)
            lvis_dt = LVISResults(lvis_gt, pred_det_results_path)
            lvis_eval = LVISEval(lvis_gt, lvis_dt, 'bbox')
            lvis_eval.params.imgIds = self.img_ids
            lvis_eval.params.catIds = list(
                self.dataset_meta['categories'].keys())
            lvis_eval.evaluate()
            lvis_eval.accumulate()
            lvis_eval.summarize()
            lvis_eval.print_results()

            lvis_results = lvis_eval.get_results()
            for k, v in lvis_results.items():
                if k.startswith('AP'):
                    key = '{}_{}'.format('bbox', k)
                    val = float('{:.3f}'.format(float(v)))
                    eval_results[key] = val
            tmp_dir.cleanup()
        return eval_results

    def format_meta(self, parts_meta: Tuple[dict]) -> dict:
        """Gather all meta info from self.results."""
        all_seq_vids_info = []
        all_seq_imgs_info = []
        all_seq_tracks_info = []
        for _seq_info in parts_meta:
            all_seq_vids_info.extend(_seq_info['videos'])
            all_seq_imgs_info.extend(_seq_info['images'])
            all_seq_tracks_info.extend(_seq_info['tracks'])

        # update tao_meta_info
        tao_meta_info = dict(
            videos=all_seq_vids_info,
            images=all_seq_imgs_info,
            tracks=all_seq_tracks_info)

        return tao_meta_info

    def format_gts(self, gts: Tuple[List], tao_meta_info: dict) -> dict:
        """Gather all ground-truth from self.results."""
        categories = list(self.dataset_meta['categories'].values())
        for img_info in tao_meta_info['images']:
            self.img_ids.append(img_info['id'])
        gt_results = dict(
            info=dict(),
            images=tao_meta_info['images'],
            categories=categories,
            videos=tao_meta_info['videos'],
            annotations=[],
            tracks=tao_meta_info['tracks'])

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
        max_track_id = 0
        for pred_result in preds:
            # update track id
            if 'track_id' in pred_result[0]:
                track_ids = []
                for ins_info in pred_result:
                    track_ids.append(ins_info['track_id'])
                    ins_info['track_id'] += max_track_id
                track_ids = list(set(track_ids))
                max_track_id += max(track_ids) + 1

            pred_results.extend(pred_result)
        return pred_results

    def _format_one_video_preds(self, pred_dicts: Tuple[dict],
                                tao_meta_info: Dict) -> Tuple[List, List]:
        """Convert the annotation to the format of YouTube-VIS.

        This operation is to make it easier to use the official eval API.

        Args:
            pred_dicts (Tuple[dict]): Prediction of the dataset.
            tao_meta_info (dict): A dict containing videos and images
                information of TAO.

        Returns:
            List: The formatted predictions.
        """
        # Collate preds scatters (tuple of dict to dict of list)
        preds = defaultdict(list)
        cat_ids = list(self.dataset_meta['categories'].keys())
        for pred in pred_dicts:
            for key in pred.keys():
                preds[key].append(pred[key])

        vid_infos = tao_meta_info['videos']
        track_json_results = []
        det_json_results = []
        video_id = vid_infos[-1]['id']

        for img_id, bboxes, scores, labels, ins_ids in zip(
                preds['img_id'], preds['track_bboxes'], preds['track_scores'],
                preds['track_labels'], preds['track_instances_id']):
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
                    category_id=cat_ids[label],
                    track_id=ins_id,
                    video_id=video_id)
                track_json_results.append(data)

        for img_id, bboxes, scores, labels in zip(preds['img_id'],
                                                  preds['det_bboxes'],
                                                  preds['det_scores'],
                                                  preds['det_labels']):
            for bbox, score, label in zip(bboxes, scores, labels):
                data = dict(
                    image_id=img_id,
                    bbox=[
                        bbox[0],
                        bbox[1],
                        bbox[2] - bbox[0],
                        bbox[3] - bbox[1],
                    ],
                    score=score,
                    category_id=cat_ids[label],
                    video_id=video_id)
                det_json_results.append(data)

        return track_json_results, det_json_results

    def _format_one_video_gts(self,
                              gt_dicts: Tuple[dict]) -> Tuple[list, dict]:
        """Convert the annotation to the format of YouTube-VIS.

        This operation is to make it easier to use the official eval API.

        Args:
            gt_dicts (Tuple[dict]): Ground truth of the dataset.

        Returns:
            Tuple[list, dict]: The formatted gts and a dict containing videos
            and images information of TAO.
        """
        video_infos = []
        image_infos = []
        track_infos = []
        annotations = []
        instance_flag = dict()  # flag the ins_id is used or not
        tao_meta_info = defaultdict(list)
        cat_ids = list(self.dataset_meta['categories'].keys())

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
                    category_id=cat_ids[label],
                    area=coco_bbox[2] * coco_bbox[3])
                if not instance_flag.get(instance_id, False):
                    track_info = dict(
                        id=instance_id,
                        category_id=cat_ids[label],
                        video_id=video_id)
                    track_infos.append(track_info)
                    instance_flag[instance_id] = True
                annotations.append(annotation)

        # update tao meta info
        tao_meta_info['images'].extend(image_infos)
        tao_meta_info['videos'].extend(video_infos)
        tao_meta_info['tracks'].extend(track_infos)

        return annotations, tao_meta_info

    def save_pred_results(self, pred_results: List, res_type: str) -> None:
        """Save the results to a zip file.

        Args:
            pred_results (list): Testing results of the
                dataset.
            res_type (str): The type of testing results, track or detection.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = self.outfile_prefix
        mmengine.dump(pred_results, f'{outfile_prefix}_{res_type}.json')

        logger.info(f'save the results to {outfile_prefix}_{res_type}.json')
