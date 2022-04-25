# Copyright (c) OpenMMLab. All rights reserved.
import contextlib
import io
from collections import OrderedDict

from mmcv.utils import print_log

from mmtrack.core.utils import YTVIS, YTVISeval


def eval_vis(json_results, vis_results, logger=None):
    """Evaluation on VIS metrics.

    Args:
        json_results (dict(list[dict])): Testing results of the VIS dataset.
        vis_results (dict(list[dict])): The annotation in the format
                of YouTube-VIS.
        logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.

    Returns:
        dict[str, float]: Evaluation results.
    """
    ytvis = YTVIS(vis_results)

    if len(ytvis.anns) == 0:
        print('Annotations does not exist')
        return

    ytvis_dets = ytvis.loadRes(json_results)
    vid_ids = ytvis.getVidIds()

    iou_type = metric = 'segm'
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
