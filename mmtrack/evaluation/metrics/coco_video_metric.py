# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

from mmdet.evaluation import CocoMetric
from mmdet.structures.mask import encode_mask_results

from mmtrack.registry import METRICS


@METRICS.register_module()
class CocoVideoMetric(CocoMetric):
    """COCO evaluation metric.

    Evaluate AR, AP, and mAP for detection tasks including proposal/box
    detection and instance segmentation. Please refer to
    https://cocodataset.org/#detection-eval for more details.

    dist_collect_mode (str, optional): The method of concatenating the
            collected synchronization results. This depends on how the
            distributed data is split. Currently only 'unzip' and 'cat' are
            supported. For samplers in MMTrakcking, 'cat' should
            be used. Defaults to 'cat'.
    dist_backend (str, optional): The name of the distributed communication
        backend, you can get all the backend names through
        ``mmeval.core.list_all_backends()``. Defaults to 'torch_cuda'.
    """

    def __init__(self,
                 dist_collect_mode='cat',
                 dist_backend='torch_cuda',
                 **kwargs) -> None:
        super().__init__(
            dist_collect_mode=dist_collect_mode,
            dist_backend=dist_backend,
            **kwargs)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Note that we only modify ``pred['pred_instances']`` in ``CocoMetric``
        to ``pred['pred_det_instances']`` here.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        predictions, groundtruths = [], []
        for data_sample in data_samples:
            pred = dict()
            pred_instances = data_sample['pred_det_instances']
            pred['img_id'] = data_sample['img_id']
            pred['bboxes'] = pred_instances['bboxes'].cpu().numpy()
            pred['scores'] = pred_instances['scores'].cpu().numpy()
            pred['labels'] = pred_instances['labels'].cpu().numpy()
            if 'masks' in pred_instances:
                pred['masks'] = encode_mask_results(
                    pred_instances['masks'].detach().cpu().numpy())
            # some detectors use different scores for bbox and mask
            if 'mask_scores' in pred_instances:
                pred['mask_scores'] = \
                    pred_instances['mask_scores'].cpu().numpy()
            predictions.append(pred)

            # parse gt
            if self._coco_api is None:
                ann = self.add_gt(data_sample)
            else:
                ann = dict()
            groundtruths.append(ann)

        self.add(predictions, groundtruths)
