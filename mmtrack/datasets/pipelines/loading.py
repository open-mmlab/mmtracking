from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile

from mmtrack.core import restore_result


@PIPELINES.register_module()
class LoadMultiImagesFromFile(LoadImageFromFile):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, results):
        outs = []
        for _results in results:
            _results = super().__call__(_results)
            outs.append(_results)
        return outs


@PIPELINES.register_module()
class SeqLoadAnnotations(LoadAnnotations):

    def __init__(self, with_track=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.with_track = with_track

    def _load_track(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded label annotations.
        """

        results['gt_instance_ids'] = results['ann_info']['instance_ids'].copy()

        return results

    def __call__(self, results):
        outs = []
        for _results in results:
            _results = super().__call__(_results)
            if self.with_track:
                _results = self._load_track(_results)
            outs.append(_results)
        return outs


@PIPELINES.register_module()
class LoadDetections(object):

    def __call__(self, results):
        detections = results['detections']
        bboxes, labels = restore_result(detections)
        results['public_bboxes'] = bboxes
        results['public_labels'] = labels
        results['bbox_fields'].append('public_bboxes')
        return results
