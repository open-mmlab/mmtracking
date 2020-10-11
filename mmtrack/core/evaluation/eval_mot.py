import time
from multiprocessing import Pool

import motmetrics as mm
import numpy as np
import pandas as pd
from mmcv.utils import print_log
from mmdet.core import bbox2result
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from motmetrics.math_util import quiet_divide

from ..track import track2result

METRIC_MAPS = {
    'mota': 'MOTA',
    'motp': 'MOTP',
    'idf1': 'IDF1',
    'num_switches': 'IDSw.',
    'num_misses': 'FN',
    'num_false_positives': 'FP',
    'mostly_tracked': 'MT',
    'partially_tracked': 'PT',
    'mostly_lost': 'ML'
}


def eval_single_video(results,
                      gts,
                      iou_thr=0.5,
                      ignore_iof_thr=0.5,
                      ignore_by_classes=False):
    num_classes = len(results[0])
    accs = [mm.MOTAccumulator(auto_id=True) for i in range(num_classes)]
    for result, gt in zip(results, gts):
        if ignore_by_classes:
            gt_ignore = bbox2result(gt['bboxes_ignore'], gt['labels_ignore'],
                                    num_classes)
        else:
            gt_ignore = [gt['bboxes_ignore'] for i in range(num_classes)]
        gt = track2result(gt['bboxes'], gt['labels'], gt['instance_ids'],
                          num_classes)
        for i in range(num_classes):
            gt_ids, gt_bboxes = gt[i][:, 0].astype(np.int), gt[i][:, 1:]
            pred_ids, pred_bboxes = result[i][:, 0].astype(
                np.int), result[i][:, 1:-1]
            if gt_ignore[i].shape[0] > 0:
                iofs = bbox_overlaps(pred_bboxes, gt_ignore[i], mode='iof')
                valid_inds = (iofs < ignore_iof_thr).all(axis=1)
                pred_ids = pred_ids[valid_inds]
                pred_bboxes = pred_bboxes[valid_inds]
            distances = mm.distances.iou_matrix(
                gt_bboxes, pred_bboxes, max_iou=1 - iou_thr)
            accs[i].update(gt_ids, pred_ids, distances)
    return accs


# TODO: polish this function
def _aggregate_eval_results(summary, metrics, classes):
    classes += ['OVERALL']
    all_summary = pd.DataFrame(columns=metrics)
    for cls in classes:
        s = summary[summary.index.str.startswith(
            str(cls))] if cls != 'OVERALL' else summary
        sum_results = s.sum()
        results = []
        for metric in metrics:
            # TODO: check the efficiency in pymotmetrics
            if metric == 'mota':
                result = 1. - quiet_divide(
                    sum_results['num_misses'] + sum_results['num_switches'] +
                    sum_results['num_false_positives'],
                    sum_results['num_objects'])
            elif metric == 'motp':
                result = quiet_divide((s['motp'] * s['num_detections']).sum(),
                                      sum_results['num_detections'])
            elif metric == 'idf1':
                result = quiet_divide(
                    2 * sum_results['idtp'], sum_results['num_objects'] +
                    sum_results['num_predictions'])
            else:
                result = sum_results[metric]
            results.append(result)
        all_summary.loc[cls] = results
    all_summary['motp'] = 1 - all_summary['motp']

    cls_summary = all_summary[~all_summary.index.str.startswith('OVERALL'
                                                                )].copy()
    cls_average = []
    avg_results = cls_summary.fillna(0).mean()
    sum_results = cls_summary.sum()
    for metric in metrics:
        if metric in ['mota', 'motp', 'idf1']:
            cls_average.append(avg_results[metric])
        else:
            cls_average.append(sum_results[metric])
    all_summary.loc['AVERAGE'] = cls_average
    dtypes = [
        'float' if m in ['mota', 'motp', 'idf1'] else 'int' for m in metrics
    ]
    dtypes = {m: d for m, d in zip(metrics, dtypes)}
    all_summary = all_summary.astype(dtypes)
    return all_summary


def eval_mot(results,
             annotations,
             logger=None,
             classes=None,
             iou_thr=0.5,
             ignore_iof_thr=0.5,
             ignore_by_classes=False,
             nproc=4):
    t = time.time()
    print_log('Evaluate CLEAR MOT metrics...', logger)
    gts = annotations.copy()
    if classes is None:
        num_classes = len(results[0])
        classes = [i + 1 for i in range(num_classes)]
    else:
        if isinstance(classes, tuple):
            classes = list(classes)
    assert len(results) == len(gts)

    print_log('Obtain results for each video...', logger)
    pool = Pool(nproc)
    results = pool.starmap(
        eval_single_video,
        zip(results, gts, [iou_thr for _ in range(len(gts))],
            [ignore_iof_thr for _ in range(len(gts))],
            [ignore_by_classes for _ in range(len(gts))]))
    pool.close()

    names, accs = [], []
    for video_ind, accs in enumerate(results):
        for i, acc in enumerate(accs):
            name = f'{classes[i]}_{video_ind}'
            if acc._events == 0:
                continue
            names.append(name)
            accs.append(acc)

    mh = mm.metrics.create()
    summary = mh.compute_many(
        accs,
        metrics=[
            'num_objects', 'num_predictions', 'num_detections', 'num_misses',
            'num_false_positives', 'num_switches', 'mostly_tracked',
            'partially_tracked', 'mostly_lost', 'idtp', 'motp'
        ],
        names=names,
        generate_overall=False)

    print_log('Aggregating...', logger)
    summary = _aggregate_eval_results(summary, METRIC_MAPS.keys(), classes)
    strsummary = mm.io.render_summary(
        summary, formatters=mh.formatters, namemap=METRIC_MAPS)
    print_log(strsummary, logger)
    print_log(f'Evaluation finishes with {(time.time() - t):.2f} s.', logger)

    summary = summary.to_dict()
    out = {METRIC_MAPS[k]: v['OVERALL'] for k, v in summary.items()}
    for k, v in out.items():
        out[k] = float(f'{(v):.3f}') if isinstance(v, float) else int(f'{v}')
    for m in ['OVERALL', 'AVERAGE']:
        out[f'track_{m}_copypaste'] = ''
        for k in METRIC_MAPS.keys():
            v = summary[k][m]
            v = f'{(v):.3f} ' if isinstance(v, float) else f'{v} '
            out[f'track_{m}_copypaste'] += v
    return out
