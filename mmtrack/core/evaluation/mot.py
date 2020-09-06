import time
from collections import defaultdict

import motmetrics as mm
import numpy as np
import pandas as pd
from motmetrics.lap import linear_sum_assignment
from motmetrics.math_util import quiet_divide

# note that there is no +1


def xyxy2xywh(bbox):
    return [
        bbox[0],
        bbox[1],
        bbox[2] - bbox[0],
        bbox[3] - bbox[1],
    ]


# 1: person, 2: vehicle, 3: bike
super_category_map = {1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 2}


def intersection_over_area(preds, gts):
    """Returns the intersection over the area of the predicted box."""
    out = np.zeros((len(preds), len(gts)))
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            x1, x2 = max(p[0], g[0]), min(p[0] + p[2], g[0] + g[2])
            y1, y2 = max(p[1], g[1]), min(p[1] + p[3], g[1] + g[3])
            out[i][j] = max(x2 - x1, 0) * max(y2 - y1, 0) / float(p[2] * p[3])

    return out


def preprocessResult(res, anns, cats_mapping, crowd_ioa_thr=0.5):
    """Preprocesses data for utils.CLEAR_MOT_M.

    Returns a subset of the predictions.
    """
    # pylint: disable=too-many-locals

    # fast indexing
    annsByAttr = defaultdict(lambda: defaultdict(list))

    for i, bbox in enumerate(anns['annotations']):
        annsByAttr[bbox['image_id']][cats_mapping[bbox['category_id']]].append(
            i)

    dropped_gt_ids = set()
    dropped_gts = []
    print('Results before drop:', sum([len(i) for i in res]))
    # match
    for (r, img) in zip(res, anns['images']):
        anns_in_frame = [
            anns['annotations'][i] for v in annsByAttr[img['id']].values()
            for i in v
        ]
        gt_bboxes = [a['bbox'] for a in anns_in_frame if not a['iscrowd']]
        res_bboxes = [xyxy2xywh(v['bbox'][:-1]) for v in r.values()]
        res_ids = list(r.keys())

        dropped_pred = []

        # drop preds that match with ignored labels
        dist = mm.distances.iou_matrix(gt_bboxes, res_bboxes, max_iou=0.5)
        le, ri = linear_sum_assignment(dist)

        ignore_gt = [
            a.get('ignore', False) for a in anns_in_frame if not a['iscrowd']
        ]
        fp_ids = set(res_ids)
        for i, j in zip(le, ri):
            if not np.isfinite(dist[i, j]):
                continue
            fp_ids.remove(res_ids[j])
            if ignore_gt[i]:
                # remove from results
                dropped_gt_ids.add(anns_in_frame[i]['id'])
                dropped_pred.append(res_ids[j])
                dropped_gts.append(i)

        # drop fps that fall in crowd regions
        crowd_gt_labels = [a['bbox'] for a in anns_in_frame if a['iscrowd']]

        if len(crowd_gt_labels) > 0 and len(fp_ids) > 0:
            ioas = np.max(
                intersection_over_area(
                    [xyxy2xywh(r[k]['bbox'][:-1]) for k in fp_ids],
                    crowd_gt_labels),
                axis=1)
            for i, ioa in zip(fp_ids, ioas):
                if ioa > crowd_ioa_thr:
                    dropped_pred.append(i)

        for p in dropped_pred:
            del r[p]

    print('Results after drop:', sum([len(i) for i in res]))


def aggregate_eval_results(summary,
                           metrics,
                           cats,
                           mh,
                           generate_overall=True,
                           class_average=False):
    if generate_overall and not class_average:
        cats.append('OVERALL')
    new_summary = pd.DataFrame(columns=metrics)
    for cat in cats:
        s = summary[summary.index.str.startswith(
            str(cat))] if cat != 'OVERALL' else summary
        res_sum = s.sum()
        new_res = []
        for metric in metrics:
            if metric == 'mota':
                res = 1. - quiet_divide(
                    res_sum['num_misses'] + res_sum['num_switches'] +
                    res_sum['num_false_positives'], res_sum['num_objects'])
            elif metric == 'motp':
                res = quiet_divide((s['motp'] * s['num_detections']).sum(),
                                   res_sum['num_detections'])
            elif metric == 'idf1':
                res = quiet_divide(
                    2 * res_sum['idtp'],
                    res_sum['num_objects'] + res_sum['num_predictions'])
            else:
                res = res_sum[metric]
            new_res.append(res)
        new_summary.loc[cat] = new_res

    new_summary['motp'] = (1 - new_summary['motp']) * 100

    if generate_overall and class_average:
        new_res = []
        res_average = new_summary.fillna(0).mean()
        res_sum = new_summary.sum()
        for metric in metrics:
            if metric in ['mota', 'motp', 'idf1']:
                new_res.append(res_average[metric])
            else:
                new_res.append(res_sum[metric])
        new_summary.loc['OVERALL'] = new_res

    dtypes = [
        'float' if m in ['mota', 'motp', 'idf1'] else 'int' for m in metrics
    ]
    dtypes = {m: d for m, d in zip(metrics, dtypes)}
    new_summary = new_summary.astype(dtypes)

    strsummary = mm.io.render_summary(
        new_summary,
        formatters=mh.formatters,
        namemap={
            'mostly_tracked': 'MT',
            'mostly_lost': 'ML',
            'num_false_positives': 'FP',
            'num_misses': 'FN',
            'num_switches': 'IDs',
            'mota': 'MOTA',
            'motp': 'MOTP',
            'idf1': 'IDF1'
        })
    print(strsummary)
    return new_summary


def eval_mot(anns, all_results, split_camera=False, class_average=False):
    print('Evaluating BDD Results...')
    assert len(all_results) == len(anns['images'])
    t = time.time()

    cats_mapping = {k['id']: k['id'] for k in anns['categories']}

    preprocessResult(all_results, anns, cats_mapping)
    anns['annotations'] = [
        a for a in anns['annotations']
        if not (a['iscrowd'] or a.get('ignore', False))
    ]

    # fast indexing
    annsByAttr = defaultdict(lambda: defaultdict(list))

    for i, bbox in enumerate(anns['annotations']):
        annsByAttr[bbox['image_id']][cats_mapping[bbox['category_id']]].append(
            i)

    track_acc = defaultdict(lambda: defaultdict())
    global_instance_id = 0
    num_instances = 0
    cat_ids = np.unique(list(cats_mapping.values()))
    video_camera_mapping = dict()
    for cat_id in cat_ids:
        for video in anns['videos']:
            track_acc[cat_id][video['id']] = mm.MOTAccumulator(auto_id=True)
            if split_camera:
                video_camera_mapping[video['id']] = video['camera_id']

    for img, results in zip(anns['images'], all_results):
        img_id = img['id']

        if img['frame_id'] == 0:
            global_instance_id += num_instances
        if len(list(results.keys())) > 0:
            num_instances = max([int(k) for k in results.keys()]) + 1

        pred_bboxes, pred_ids = defaultdict(list), defaultdict(list)
        for instance_id, result in results.items():
            _bbox = xyxy2xywh(result['bbox'])
            _cat = cats_mapping[result['label'] + 1]
            pred_bboxes[_cat].append(_bbox)
            instance_id = int(instance_id) + global_instance_id
            pred_ids[_cat].append(instance_id)

        gt_bboxes, gt_ids = defaultdict(list), defaultdict(list)
        for cat_id in cat_ids:
            for i in annsByAttr[img_id][cat_id]:
                ann = anns['annotations'][i]
                gt_bboxes[cat_id].append(ann['bbox'])
                gt_ids[cat_id].append(ann['instance_id'])
            distances = mm.distances.iou_matrix(
                gt_bboxes[cat_id], pred_bboxes[cat_id], max_iou=0.5)
            track_acc[cat_id][img['video_id']].update(gt_ids[cat_id],
                                                      pred_ids[cat_id],
                                                      distances)

    # eval for track
    print('Generating matchings and summary...')
    empty_cat = []
    for cat, video_track_acc in track_acc.items():
        for vid, v in video_track_acc.items():
            if len(v._events) == 0:
                empty_cat.append([cat, vid])
    for cat, vid in empty_cat:
        track_acc[cat].pop(vid)

    names, acc = [], []
    for cat, video_track_acc in track_acc.items():
        for vid, v in video_track_acc.items():
            name = '{}_{}'.format(cat, vid)
            if split_camera:
                name += '_{}'.format(video_camera_mapping[vid])
            names.append(name)
            acc.append(v)

    metrics = [
        'mota', 'motp', 'num_misses', 'num_false_positives', 'num_switches',
        'mostly_tracked', 'mostly_lost', 'idf1'
    ]

    print('Evaluating box tracking...')
    mh = mm.metrics.create()
    summary = mh.compute_many(
        acc,
        metrics=[
            'num_objects', 'motp', 'num_detections', 'num_misses',
            'num_false_positives', 'num_switches', 'mostly_tracked',
            'mostly_lost', 'idtp', 'num_predictions'
        ],
        names=names,
        generate_overall=False)
    if split_camera:
        summary['camera_id'] = summary.index.str.split('_').str[-1]
        for camera_id, summary_ in summary.groupby('camera_id'):
            print('\nEvaluating camera ID: ', camera_id)
            aggregate_eval_results(
                summary_,
                metrics,
                list(track_acc.keys()),
                mh,
                generate_overall=True,
                class_average=class_average)

    print('\nEvaluating overall results...')
    summary = aggregate_eval_results(
        summary,
        metrics,
        list(track_acc.keys()),
        mh,
        generate_overall=True,
        class_average=class_average)

    print('Evaluation finsihes with {:.2f} s'.format(time.time() - t))

    out = {k: v for k, v in summary.to_dict().items()}
    return out
