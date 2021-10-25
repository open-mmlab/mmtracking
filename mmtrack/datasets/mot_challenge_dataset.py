# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import tempfile

import mmcv
import motmetrics as mm
import numpy as np
from mmcv.utils import print_log
from mmdet.core import eval_map
from mmdet.datasets import DATASETS

from mmtrack.core import results2outs
from .coco_video_dataset import CocoVideoDataset


@DATASETS.register_module()
class MOTChallengeDataset(CocoVideoDataset):
    """Dataset for MOTChallenge.

    Args:
        visibility_thr (float, optional): The minimum visibility
            for the objects during training. Default to -1.
        detection_file (str, optional): The path of the public
            detection file. Default to None.
    """

    CLASSES = ('pedestrian', )

    def __init__(self,
                 visibility_thr=-1,
                 detection_file=None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.visibility_thr = visibility_thr
        self.detections = self.load_detections(detection_file)

    def load_detections(self, detection_file=None):
        """Load public detections."""
        # support detections in three formats
        # 1. MMDet: [img_1, img_2, ...]
        # 2. MMTrack: dict(det_bboxes=[img_1, img_2, ...])
        # 3. Public:
        #    1) dict(img1_name: [], img2_name: [], ...)
        #    2) dict(det_bboxes=dict(img1_name: [], img2_name: [], ...))
        # return as a dict or a list
        if detection_file is not None:
            detections = mmcv.load(detection_file)
            if isinstance(detections, dict):
                # results from mmtrack
                if 'det_bboxes' in detections:
                    detections = detections['det_bboxes']
            else:
                # results from mmdet
                if not isinstance(detections, list):
                    raise TypeError('detections must be a dict or a list.')
            return detections
        else:
            return None

    def prepare_results(self, img_info):
        """Prepare results for image (e.g. the annotation information, ...)."""
        results = super().prepare_results(img_info)
        if self.detections is not None:
            if isinstance(self.detections, dict):
                indice = img_info['file_name']
            elif isinstance(self.detections, list):
                indice = self.img_ids.index(img_info['id'])
            results['detections'] = self.detections[indice]
        return results

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
            labels, masks, seg_map. "masks" are raw annotations and not \
            decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_instance_ids = []

        for i, ann in enumerate(ann_info):
            if (not self.test_mode) and (ann['visibility'] <
                                         self.visibility_thr):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('ignore', False) or ann.get('iscrowd', False):
                # note: normally no `iscrowd` for MOT17Dataset
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_instance_ids.append(ann['instance_id'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
            gt_instance_ids = np.array(gt_instance_ids, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
            gt_instance_ids = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            instance_ids=gt_instance_ids)

        return ann

    def format_results(self, results, resfile_path=None, metrics=['track']):
        """Format the results to txts (standard format for MOT Challenge).

        Args:
            results (dict(list[ndarray])): Testing results of the dataset.
            resfile_path (str, optional): Path to save the formatted results.
                Defaults to None.
            metrics (list[str], optional): The results of the specific metrics
                will be formatted.. Defaults to ['track'].

        Returns:
            tuple: (resfiles, names, tmp_dir), resfiles is a dict containing
            the filepaths, names is a list containing the name of the
            videos, tmp_dir is the temporal directory created for saving
            files.
        """
        assert isinstance(results, dict), 'results must be a dict.'
        if resfile_path is None:
            tmp_dir = tempfile.TemporaryDirectory()
            resfile_path = tmp_dir.name
        else:
            tmp_dir = None
            if osp.exists(resfile_path):
                print_log('remove previous results.', self.logger)
                import shutil
                shutil.rmtree(resfile_path)

        resfiles = dict()
        for metric in metrics:
            resfiles[metric] = osp.join(resfile_path, metric)
            os.makedirs(resfiles[metric], exist_ok=True)

        inds = [i for i, _ in enumerate(self.data_infos) if _['frame_id'] == 0]
        num_vids = len(inds)
        assert num_vids == len(self.vid_ids)
        inds.append(len(self.data_infos))
        vid_infos = self.coco.load_vids(self.vid_ids)
        names = [_['name'] for _ in vid_infos]

        for i in range(num_vids):
            for metric in metrics:
                formatter = getattr(self, f'format_{metric}_results')
                formatter(results[f'{metric}_bboxes'][inds[i]:inds[i + 1]],
                          self.data_infos[inds[i]:inds[i + 1]],
                          f'{resfiles[metric]}/{names[i]}.txt')

        return resfiles, names, tmp_dir

    def format_track_results(self, results, infos, resfile):
        """Format tracking results."""
        with open(resfile, 'wt') as f:
            for res, info in zip(results, infos):
                if 'mot_frame_id' in info:
                    frame = info['mot_frame_id']
                else:
                    frame = info['frame_id'] + 1

                outs_track = results2outs(bbox_results=res)
                for bbox, label, id in zip(outs_track['bboxes'],
                                           outs_track['labels'],
                                           outs_track['ids']):
                    x1, y1, x2, y2, conf = bbox
                    f.writelines(
                        f'{frame},{id},{x1:.3f},{y1:.3f},{(x2-x1):.3f},' +
                        f'{(y2-y1):.3f},{conf:.3f},-1,-1,-1\n')

    def format_det_results(self, results, infos, resfile):
        """Format detection results."""
        with open(resfile, 'wt') as f:
            for res, info in zip(results, infos):
                if 'mot_frame_id' in info:
                    frame = info['mot_frame_id']
                else:
                    frame = info['frame_id'] + 1

                outs_det = results2outs(bbox_results=res)
                for bbox, label in zip(outs_det['bboxes'], outs_det['labels']):
                    x1, y1, x2, y2, conf = bbox
                    f.writelines(
                        f'{frame},-1,{x1:.3f},{y1:.3f},{(x2-x1):.3f},' +
                        f'{(y2-y1):.3f},{conf:.3f}\n')
            f.close()

    def evaluate(self,
                 results,
                 metric='track',
                 logger=None,
                 resfile_path=None,
                 bbox_iou_thr=0.5,
                 track_iou_thr=0.5):
        """Evaluation in MOT Challenge.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'track'. Defaults to 'track'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            resfile_path (str, optional): Path to save the formatted results.
                Defaults to None.
            bbox_iou_thr (float, optional): IoU threshold for detection
                evaluation. Defaults to 0.5.
            track_iou_thr (float, optional): IoU threshold for tracking
                evaluation.. Defaults to 0.5.

        Returns:
            dict[str, float]: MOTChallenge style evaluation metric.
        """
        eval_results = dict()
        if isinstance(metric, list):
            metrics = metric
        elif isinstance(metric, str):
            metrics = [metric]
        else:
            raise TypeError('metric must be a list or a str.')
        allowed_metrics = ['det', 'track']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported.')

        if 'track' in metrics:
            resfiles, names, tmp_dir = self.format_results(
                results, resfile_path, metrics)
            print_log('Evaluate CLEAR MOT results.', logger=logger)
            distth = 1 - track_iou_thr

            accs = []
            for name in names:
                if 'half-train' in self.ann_file:
                    gt_file = osp.join(self.img_prefix,
                                       f'{name}/gt/gt_half-train.txt')
                elif 'half-val' in self.ann_file:
                    gt_file = osp.join(self.img_prefix,
                                       f'{name}/gt/gt_half-val.txt')
                else:
                    gt_file = osp.join(self.img_prefix, f'{name}/gt/gt.txt')
                res_file = osp.join(resfiles['track'], f'{name}.txt')
                gt = mm.io.loadtxt(gt_file)
                res = mm.io.loadtxt(res_file)
                ini_file = osp.join(self.img_prefix, f'{name}/seqinfo.ini')
                if osp.exists(ini_file) and 'MOT15' not in self.img_prefix:
                    acc, ana = mm.utils.CLEAR_MOT_M(
                        gt, res, ini_file, distth=distth)
                else:
                    acc = mm.utils.compare_to_groundtruth(
                        gt, res, distth=distth)
                accs.append(acc)

            mh = mm.metrics.create()
            summary = mh.compute_many(
                accs,
                names=names,
                metrics=mm.metrics.motchallenge_metrics,
                generate_overall=True)
            str_summary = mm.io.render_summary(
                summary,
                formatters=mh.formatters,
                namemap=mm.io.motchallenge_metric_names)
            print(str_summary)

            eval_results.update({
                mm.io.motchallenge_metric_names[k]: v['OVERALL']
                for k, v in summary.to_dict().items()
            })

            if tmp_dir is not None:
                tmp_dir.cleanup()

        if 'det' in metrics:
            if isinstance(results, dict):
                bbox_results = results['det_bboxes']
            elif isinstance(results, list):
                bbox_results = results
            else:
                raise TypeError('results must be a dict or a list.')
            annotations = [self.get_ann_info(info) for info in self.data_infos]
            mean_ap, _ = eval_map(
                bbox_results,
                annotations,
                iou_thr=bbox_iou_thr,
                dataset=self.CLASSES,
                logger=logger)
            eval_results['mAP'] = mean_ap

        for k, v in eval_results.items():
            if isinstance(v, float):
                eval_results[k] = float(f'{(v):.3f}')

        return eval_results
