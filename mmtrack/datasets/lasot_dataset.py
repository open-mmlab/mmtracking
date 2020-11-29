import numpy as np
from mmcv.utils import print_log
from mmdet.datasets import DATASETS

from mmtrack.core.evaluation import eval_ope_benchmark
from .coco_video_dataset import CocoVideoDataset


@DATASETS.register_module()
class LaSOTDataset(CocoVideoDataset):

    CLASSES = (0, )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _parse_ann_info(self, img_info, ann_info):
        gt_bboxes = np.array(ann_info[0]['bbox'], dtype=np.float32)
        # convert [x1, y1, w, h] to [x1, y1, x2, y2]
        gt_bboxes[2] += gt_bboxes[0]
        gt_bboxes[3] += gt_bboxes[1]
        gt_labels = np.array(self.cat2label[ann_info[0]['category_id']])
        ignore = ann_info[0]['full_occlusion'] or ann_info[0]['out_of_view']
        ann = dict(bboxes=gt_bboxes, labels=gt_labels, ignore=ignore)
        return ann

    def evaluate(self, results, metric=['track'], logger=None):
        if isinstance(metric, list):
            metrics = metric
        elif isinstance(metric, str):
            metrics = [metric]
        else:
            raise TypeError('metric must be a list or a str.')
        allowed_metrics = ['track']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported.')

        eval_results = dict()
        if 'track' in metrics:
            assert len(self.data_infos) == len(results['bbox'])
            print_log('Evaluate OPE Benchmark...', logger=logger)
            inds = [
                i for i, _ in enumerate(self.data_infos) if _['frame_id'] == 0
            ]
            num_vids = len(inds)
            inds.append(len(self.data_infos))

            track_results = [
                results['bbox'][inds[i]:inds[i + 1]] for i in range(num_vids)
            ]

            ann_infos = [self.get_ann_info(_) for _ in self.data_infos]
            ann_infos = [
                ann_infos[inds[i]:inds[i + 1]] for i in range(num_vids)
            ]
            track_eval_results = eval_ope_benchmark(
                results=track_results, annotations=ann_infos)
            eval_results.update(track_eval_results)
            print_log(eval_results, logger=logger)

        return eval_results
