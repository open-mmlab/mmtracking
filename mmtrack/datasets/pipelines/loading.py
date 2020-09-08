from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile


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

    def __init__(self, with_ins_id=False, *args, **kwargs):
        # TODO: name
        super().__init__(*args, **kwargs)
        self.with_ins_id = with_ins_id

    def _load_ins_ids(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded label annotations.
        """

        results['gt_match_indices'] = results['ann_info'][
            'match_indices'].copy()

        return results

    def __call__(self, results):
        outs = []
        for _results in results:
            _results = super().__call__(_results)
            if self.with_ins_id:
                _results = self._load_ins_ids(_results)
            outs.append(_results)
        return outs
