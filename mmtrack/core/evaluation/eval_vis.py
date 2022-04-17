# Copyright (c) OpenMMLab. All rights reserved.

import contextlib
import copy
import io
from collections import OrderedDict, defaultdict

import mmcv
from mmcv.utils import print_log
from pycocotools.ytvos import YTVOS
from pycocotools.ytvoseval import YTVOSeval


class YTVIS(YTVOS):
    """Override the init of VTVOS to accept dict result.

    Args:
        json_result (dict): YouTube-VIS format json result.
    """

    def __init__(self, json_result):
        self.dataset, self.anns, self.cats, self.vids = dict(), dict(), dict(
        ), dict()
        self.vidToAnns, self.catToVids = defaultdict(list), defaultdict(list)
        if json_result is None:
            dataset = json_result
            assert type(
                dataset
            ) == dict, 'annotation file format {} not supported'.format(
                type(dataset))
            self.dataset = dataset
            self.createIndex()


def eval_vis(result_file, own_anns_file, logger):
    """Evaluation VIS metrics.

    Args:
        result_file (str): The path of json file which has been
            converted to YouTube-VIS format.
        own_anns_file (str): The path of COCO style annotation file.
        logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.

    Returns:
        dict[str, float]: Evaluation results.
    """
    ytvos = get_vis_json(own_anns_file)
    ytvos = YTVIS(ytvos)
    assert isinstance(ytvos, YTVIS)

    if len(ytvos.anns) == 0:
        print('Annotations does not exist')
        return
    assert result_file.endswith('.json')
    ytvos_dets = ytvos.loadRes(result_file)
    vid_ids = ytvos.getVidIds()

    metric = 'segm'
    iou_type = metric
    eval_results = OrderedDict()
    ytvosEval = YTVOSeval(ytvos, ytvos_dets, iou_type)
    ytvosEval.params.vidIds = vid_ids
    ytvosEval.evaluate()
    ytvosEval.accumulate()
    redirect_string = io.StringIO()
    with contextlib.redirect_stdout(redirect_string):
        ytvosEval.summarize()
    print_log('\n' + redirect_string.getvalue(), logger=logger)

    metric_items = ['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l']
    coco_metric_names = {
        'mAP': 0,
        'mAP_50': 1,
        'mAP_75': 2,
        'mAP_s': 3,
        'mAP_m': 4,
        'mAP_l': 5,
        'AR@100': 6,
        'AR@300': 7,
        'AR@1000': 8,
        'AR_s@1000': 9,
        'AR_m@1000': 10,
        'AR_l@1000': 11
    }
    for metric_item in metric_items:
        key = f'{metric}_{metric_item}'
        val = float(f'{ytvosEval.stats[coco_metric_names[metric_item]]:.3f}')
        eval_results[key] = val
    ap = ytvosEval.stats[:6]
    eval_results[f'{metric}_mAP_copypaste'] = (
        f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
        f'{ap[4]:.3f} {ap[5]:.3f}')
    return eval_results


def get_vis_json(own_anns_file):
    """Convert the annotation to the format of YouTube-VIS.

    Args:
        own_anns_file: The path of COCO style annotation file.

    Returns:
        dict: A dict with 3 keys, ``categories``, ``annotations``
            and ``videos``.
    """

    VIS = defaultdict(list)
    own_anns = mmcv.load(own_anns_file)
    VIS['categories'] = copy.deepcopy(own_anns['categories'])
    VIS['videos'] = copy.deepcopy(own_anns['videos'])

    instance_info = defaultdict(list)
    frame_id = defaultdict(list)
    len_video = defaultdict(list)
    for ann_info in own_anns['annotations']:
        instance_info[ann_info['instance_id']].append(ann_info)

    for img_info in own_anns['images']:
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
