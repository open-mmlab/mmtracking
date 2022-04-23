# Copyright (c) OpenMMLab. All rights reserved.
import contextlib
import copy
import io
from collections import OrderedDict, defaultdict

import mmcv
from mmcv.utils import print_log

from mmtrack.core.utils import YTVIS, YTVISeval


def eval_vis(json_results, vis_anns_file, logger=None):
    """Evaluation VIS metrics.

    Args:
        json_results (dict(list[dict])): Testing results of the VIS dataset.
        vis_anns_file (str): The path of COCO style annotation file.
        logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.

    Returns:
        dict[str, float]: Evaluation results.
    """
    VIS = convert_vis_fmt(vis_anns_file)
    ytvis = YTVIS(VIS)

    if len(ytvis.anns) == 0:
        print('Annotations does not exist')
        return

    ytvis_dets = ytvis.loadRes(json_results)
    vid_ids = ytvis.getVidIds()

    metric = 'segm'
    iou_type = metric
    eval_results = OrderedDict()
    ytvisEval = YTVISeval(ytvis, ytvis_dets, iou_type)
    ytvisEval.params.vidIds = vid_ids
    ytvisEval.evaluate()
    ytvisEval.accumulate()
    redirect_string = io.StringIO()
    with contextlib.redirect_stdout(redirect_string):
        ytvisEval.summarize()
    print_log('\n' + redirect_string.getvalue(), logger=logger)

    metric_items = ['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l']
    coco_metric_names = {
        'mAP': 0,
        'mAP_50': 1,
        'mAP_75': 2,
        'mAP_s': 3,
        'mAP_m': 4,
        'mAP_l': 5,
        'AR@1': 6,
        'AR@10': 7,
        'AR@100': 8,
        'AR_s@100': 9,
        'AR_m@100': 10,
        'AR_l@100': 11
    }
    for metric_item in metric_items:
        key = f'{metric}_{metric_item}'
        val = float(f'{ytvisEval.stats[coco_metric_names[metric_item]]:.3f}')
        eval_results[key] = val
    ap = ytvisEval.stats[:6]
    eval_results[f'{metric}_mAP_copypaste'] = (
        f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
        f'{ap[4]:.3f} {ap[5]:.3f}')
    return eval_results


def convert_vis_fmt(vis_anns_file):
    """Convert the annotation to the format of YouTube-VIS.

    Args:
        vis_anns_file: The path of COCO style annotation file.

    Returns:
        dict: A dict with 3 keys, ``categories``, ``annotations``
            and ``videos``.
        - | ``categories`` (dict{list[dict]}): Each list has a dict
            with 2 keys, ``id`` and ``name``.
        - | ``videos`` (dict{list[dict]}): Each list has a dict with
            4 keys of video info, ``id``, ``name``, ``width`` and ``height``.
        - | ``annotations`` (dict{list[dict]}): Each list has a dict with
            7 keys of video info, ``category_id``, ``segmentations``,
            ``bboxes``, ``video_id``, ``areas``, ``id`` and ``iscrowd``.
    """

    VIS = defaultdict(list)
    ori_anns = mmcv.load(vis_anns_file)
    VIS['categories'] = copy.deepcopy(ori_anns['categories'])
    VIS['videos'] = copy.deepcopy(ori_anns['videos'])

    instance_info = defaultdict(list)
    frame_id = defaultdict(list)
    len_video = defaultdict(list)
    for ann_info in ori_anns['annotations']:
        instance_info[ann_info['instance_id']].append(ann_info)

    for img_info in ori_anns['images']:
        frame_id[img_info['id']] = img_info['frame_id']

        len_video[img_info['video_id']] = max(1, img_info['frame_id'] + 1)
    for idx in instance_info:
        cur_video_len = len_video[instance_info[idx][0]['video_id']]
        segm = [None] * cur_video_len
        bbox = [None] * cur_video_len
        area = [None] * cur_video_len

        for ann_info in instance_info[idx]:
            segm[frame_id[ann_info['image_id']]] = ann_info['segmentation']
            bbox[frame_id[ann_info['image_id']]] = ann_info['bbox']
            area[frame_id[ann_info['image_id']]] = ann_info['area']

        instance = dict(
            category_id=instance_info[idx][0]['category_id'],
            segmentations=segm,
            bboxes=bbox,
            video_id=instance_info[idx][0]['video_id'],
            areas=area,
            id=instance_info[idx][0]['instance_id'],
            iscrowd=instance_info[idx][0]['iscrowd'])
        VIS['annotations'].append(instance)
    return dict(VIS)
