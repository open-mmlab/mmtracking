import numpy as np
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class MatchInstances(object):

    def __init__(self, skip_nomatch=True):
        self.skip_nomatch = skip_nomatch

    def _match_gts(self, instance_ids, ref_instance_ids):
        ins_ids = list(instance_ids)
        ref_ins_ids = list(ref_instance_ids)
        match_indices = np.array([
            ref_ins_ids.index(i) if (i in ref_ins_ids and i > 0) else -1
            for i in ins_ids
        ])
        ref_match_indices = np.array([
            ins_ids.index(i) if (i in ins_ids and i > 0) else -1
            for i in ref_ins_ids
        ])
        return match_indices, ref_match_indices

    def __call__(self, results):
        if len(results) != 2:
            raise NotImplementedError('Only support match 2 images now.')

        match_indices, ref_match_indices = self._match_gts(
            results[0]['gt_instance_ids'], results[1]['gt_instance_ids'])
        nomatch = (match_indices == -1).all()
        if self.skip_nomatch and nomatch:
            return None
        else:
            results[0]['gt_match_indices'] = match_indices.copy()
            results[1]['gt_match_indices'] = ref_match_indices.copy()

        return results
