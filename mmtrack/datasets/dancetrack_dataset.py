# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import tempfile

import motmetrics as mm
import numpy as np
from mmcv.utils import print_log
from mmdet.core import eval_map
from mmdet.datasets import DATASETS

from .mot_challenge_dataset import MOTChallengeDataset

try:
    import trackeval
except ImportError:
    trackeval = None


@DATASETS.register_module()
class DanceTrackDataset(MOTChallengeDataset):
    """Dataset for DanceTrack: https://github.com/DanceTrack/DanceTrack.

    Most content is inherited from MOTChallengeDataset.
    """

    def get_dataset_cfg_for_hota(self, gt_folder, tracker_folder, seqmap):
        """Get default configs for trackeval.datasets.MotChallenge2DBox.

        Args:
            gt_folder (str): the name of the GT folder
            tracker_folder (str): the name of the tracker folder
            seqmap (str): the file that contains the sequence of video names

        Returns:
            Dataset Configs for MotChallenge2DBox.
        """
        dataset_config = dict(
            # Location of GT data
            GT_FOLDER=gt_folder,
            # Trackers location
            TRACKERS_FOLDER=tracker_folder,
            # Where to save eval results
            # (if None, same as TRACKERS_FOLDER)
            OUTPUT_FOLDER=None,
            # Use 'track' as the default tracker
            TRACKERS_TO_EVAL=['track'],
            # Option values: ['pedestrian']
            CLASSES_TO_EVAL=list(self.CLASSES),
            # TrackEval does not support Dancetrack as an option,
            # we use the wrapper for MOT17 dataset
            BENCHMARK='DanceTrack',
            # Option Values: 'train', 'val', 'test'
            SPLIT_TO_EVAL='val',
            # Whether tracker input files are zipped
            INPUT_AS_ZIP=False,
            # Whether to print current config
            PRINT_CONFIG=True,
            # Whether to perform preprocessing
            # (never done for MOT15)
            DO_PREPROC=False if 'MOT15' in self.img_prefix else True,
            # Tracker files are in
            # TRACKER_FOLDER/tracker_name/TRACKER_SUB_FOLDER
            TRACKER_SUB_FOLDER='',
            # Output files are saved in
            # OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
            OUTPUT_SUB_FOLDER='',
            # Names of trackers to display
            # (if None: TRACKERS_TO_EVAL)
            TRACKER_DISPLAY_NAMES=None,
            # Where seqmaps are found
            # (if None: GT_FOLDER/seqmaps)
            SEQMAP_FOLDER=None,
            # Directly specify seqmap file
            # (if none use seqmap_folder/benchmark-split_to_eval)
            SEQMAP_FILE=seqmap,
            # If not None, specify sequences to eval
            # and their number of timesteps
            SEQ_INFO=None,
            # '{gt_folder}/{seq}/gt/gt.txt'
            GT_LOC_FORMAT='{gt_folder}/{seq}/gt/gt.txt',
            # If False, data is in GT_FOLDER/BENCHMARK-SPLIT_TO_EVAL/ and in
            # TRACKERS_FOLDER/BENCHMARK-SPLIT_TO_EVAL/tracker/
            # If True, the middle 'benchmark-split' folder is skipped for both.
            SKIP_SPLIT_FOL=True,
        )

        return dataset_config

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
        allowed_metrics = ['bbox', 'track']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported.')

        if 'track' in metrics:
            resfile_path, resfiles, names, tmp_dir = self.format_results(
                results, resfile_path, metrics)
            print_log('Evaluate CLEAR MOT results.', logger=logger)
            distth = 1 - track_iou_thr
            accs = []
            # support loading data from ceph
            local_dir = tempfile.TemporaryDirectory()

            for name in names:
                gt_file = osp.join(self.img_prefix, f'{name}/gt/gt.txt')
                res_file = osp.join(resfiles['track'], f'{name}.txt')
                # copy gt file from ceph to local temporary directory
                gt_dir_path = osp.join(local_dir.name, name, 'gt')
                os.makedirs(gt_dir_path)
                copied_gt_file = osp.join(
                    local_dir.name,
                    gt_file.replace(gt_file.split(name)[0], ''))

                f = open(copied_gt_file, 'wb')
                gt_content = self.file_client.get(gt_file)
                if hasattr(gt_content, 'tobytes'):
                    gt_content = gt_content.tobytes()
                f.write(gt_content)
                f.close()
                # copy sequence file from ceph to local temporary directory
                copied_seqinfo_path = osp.join(local_dir.name, name,
                                               'seqinfo.ini')
                f = open(copied_seqinfo_path, 'wb')
                seq_content = self.file_client.get(
                    osp.join(self.img_prefix, name, 'seqinfo.ini'))
                if hasattr(seq_content, 'tobytes'):
                    seq_content = seq_content.tobytes()
                f.write(seq_content)
                f.close()

                gt = mm.io.loadtxt(copied_gt_file)
                res = mm.io.loadtxt(res_file)
                if osp.exists(copied_seqinfo_path):
                    acc, ana = mm.utils.CLEAR_MOT_M(
                        gt, res, copied_seqinfo_path, distth=distth)
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

            if trackeval is None:
                raise ImportError(
                    'Please run'
                    'pip install git+https://github.com/JonathonLuiten/TrackEval.git'  # noqa
                    'to manually install trackeval')

            seqmap = osp.join(resfile_path, 'videoseq.txt')
            with open(seqmap, 'w') as f:
                f.write('name\n')
                for name in names:
                    f.write(name + '\n')
                f.close()

            eval_config = trackeval.Evaluator.get_default_eval_config()

            # tracker's name is set to 'track',
            # so this word needs to be splited out
            output_folder = resfiles['track'].rsplit(os.sep, 1)[0]
            dataset_config = self.get_dataset_cfg_for_hota(
                local_dir.name, output_folder, seqmap)

            evaluator = trackeval.Evaluator(eval_config)
            dataset = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
            hota_metrics = [
                trackeval.metrics.HOTA(dict(METRICS=['HOTA'], THRESHOLD=0.5))
            ]
            output_res, _ = evaluator.evaluate(dataset, hota_metrics)

            # modify HOTA results sequence according to summary list,
            # indexes of summary are sequence names and 'OVERALL'
            # while for hota they are sequence names and 'COMBINED_SEQ'
            seq_list = list(summary.index)
            seq_list.append('COMBINED_SEQ')

            hota = [
                np.average(output_res['MotChallenge2DBox']['track'][seq]
                           ['pedestrian']['HOTA']['HOTA']) for seq in seq_list
                if 'OVERALL' not in seq
            ]

            eval_results.update({
                mm.io.motchallenge_metric_names[k]: v['OVERALL']
                for k, v in summary.to_dict().items()
            })
            eval_results['HOTA'] = hota[-1]

            summary['HOTA'] = hota
            str_summary = mm.io.render_summary(
                summary,
                formatters=mh.formatters,
                namemap=mm.io.motchallenge_metric_names)
            print(str_summary)
            local_dir.cleanup()
            if tmp_dir is not None:
                tmp_dir.cleanup()

        if 'bbox' in metrics:
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
