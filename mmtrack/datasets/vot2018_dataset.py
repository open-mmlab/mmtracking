import numpy as np
from mmcv.utils import print_log
from mmdet.datasets import DATASETS

from mmtrack.core.evaluation import eval_sot_accuracy_robustness, eval_sot_eao
from .coco_video_dataset import CocoVideoDataset


@DATASETS.register_module()
class VOT2018Dataset(CocoVideoDataset):
    """VOT2018 dataset for the testing of single object tracking.

    The dataset doesn't support training mode.
    """
    CLASSES = (0, )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox annotations.

        Args:
            img_info (dict): image information.
            ann_info (list[dict]): Annotation information of an image. Each
                image only has one bbox annotation.
        Returns:
            dict: A dict containing the following keys: bboxes, labels.
            labels are not useful in SOT.
        """
        gt_bboxes = np.array(ann_info[0]['bbox'], dtype=np.float32)
        gt_labels = np.array(self.cat2label[ann_info[0]['category_id']])
        ann = dict(bboxes=gt_bboxes, labels=gt_labels)
        return ann

    def evaluate(self,
                 results,
                 metric=['track'],
                 interval=[100, 356],
                 logger=None):
        """Evaluation in VOT protocol.

        Args:
            results (dict): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'track'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
        Returns:
            dict[str, float]:
        """
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
            assert len(self.data_infos) == len(results['track_bboxes'])
            print_log('Evaluate VOT Benchmark...', logger=logger)
            inds = []
            region_bound = []
            ann_infos = []
            videos_name = []
            for i, info in enumerate(self.data_infos):
                if info['frame_id'] == 0:
                    inds.append(i)
                    region_bound.append((info['width'], info['height']))
                    video_id = info['video_id']
                    video_name = self.coco.dataset['videos'][video_id -
                                                             1]['name']
                    videos_name.append(video_name)

                ann_infos.append(self.get_ann_info(info))

            num_vids = len(inds)
            inds.append(len(self.data_infos))

            track_bboxes = []
            annotations = []
            for i in range(num_vids):
                bboxes_per_video = []
                for bbox in results['track_bboxes'][inds[i]:inds[i + 1]]:
                    if len(bbox) != 2:
                        # convert bbox format from (tl_x, tl_y, br_x, br_y) to
                        # (x, y, w, h)
                        bbox[2] -= bbox[0]
                        bbox[3] -= bbox[1]
                    bboxes_per_video.append(bbox[:-1])
                track_bboxes.append(bboxes_per_video)
                annotations.append(ann_infos[inds[i]:inds[i + 1]])

                # TODO del this piece of code
                save_dir = 'logs/vot2018/best_eao_epoch_11_bbox_results'
                import os
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)
                with open(
                        os.path.join(save_dir, f'{videos_name[i]}.txt'),
                        'w') as f:
                    for x in bboxes_per_video:
                        f.write(','.join(list(map(str, x))) + '\n')

            # anno_info is list[list]
            eao_score = eval_sot_eao(
                results=track_bboxes,
                annotations=annotations,
                region_bound=region_bound,
                interval=interval)
            eval_results.update(eao_score)

            accuracy_robustness = eval_sot_accuracy_robustness(
                results=track_bboxes,
                annotations=annotations,
                region_bound=region_bound)
            eval_results.update(accuracy_robustness)

            for k, v in eval_results.items():
                if isinstance(v, float):
                    eval_results[k] = float(f'{(v):.4f}')
            print_log(eval_results, logger=logger)

        return eval_results
