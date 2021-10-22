# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile

from mmtrack.core import results2outs


@PIPELINES.register_module()
class LoadMultiImagesFromFile(LoadImageFromFile):
    """Load multi images from file.

    Please refer to `mmdet.datasets.pipelines.loading.py:LoadImageFromFile`
    for detailed docstring.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, results):
        """Call function.

        For each dict in `results`, call the call function of
        `LoadImageFromFile` to load image.

        Args:
            results (list[dict]): List of dict from
                :obj:`mmtrack.CocoVideoDataset`.

        Returns:
            list[dict]: List of dict that contains loaded image.
        """
        outs = []
        for _results in results:
            _results = super().__call__(_results)
            outs.append(_results)
        return outs


@PIPELINES.register_module()
class SeqLoadAnnotations(LoadAnnotations):
    """Sequence load annotations.

    Please refer to `mmdet.datasets.pipelines.loading.py:LoadAnnotations`
    for detailed docstring.

    Args:
        with_track (bool): If True, load instance ids of bboxes.
    """

    def __init__(self, with_track=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.with_track = with_track

    def _load_track(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmtrack.CocoVideoDataset`.

        Returns:
            dict: The dict contains loaded label annotations.
        """

        results['gt_instance_ids'] = results['ann_info']['instance_ids'].copy()

        return results

    def __call__(self, results):
        """Call function.

        For each dict in results, call the call function of `LoadAnnotations`
        to load annotation.

        Args:
            results (list[dict]): List of dict that from
                :obj:`mmtrack.CocoVideoDataset`.

        Returns:
            list[dict]: List of dict that contains loaded annotations, such as
            bounding boxes, labels, instance ids, masks and semantic
            segmentation annotations.
        """
        outs = []
        for _results in results:
            _results = super().__call__(_results)
            if self.with_track:
                _results = self._load_track(_results)
            outs.append(_results)
        return outs


@PIPELINES.register_module()
class LoadDetections(object):
    """Load public detections from MOT benchmark.

    Args:
        results (dict): Result dict from :obj:`mmtrack.CocoVideoDataset`.
    """

    def __call__(self, results):
        outs_det = results2outs(bbox_results=results['detections'])
        bboxes = outs_det['bboxes']
        labels = outs_det['labels']

        results['public_bboxes'] = bboxes[:, :4]
        if bboxes.shape[1] > 4:
            results['public_scores'] = bboxes[:, -1]
        results['public_labels'] = labels
        results['bbox_fields'].append('public_bboxes')
        return results
