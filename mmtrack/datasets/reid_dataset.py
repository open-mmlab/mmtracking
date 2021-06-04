import torch
import mmcv
import numpy as np

from mmcls.datasets import DATASETS
from mmcls.datasets import BaseDataset


@DATASETS.register_module()
class ReIDDataset(BaseDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def load_annotations(self):
        assert isinstance(self.ann_file, str)

        data_infos = []
        with open(self.ann_file) as f:
            samples = [x.strip().split(' ') for x in f.readlines()]
            for filename, gt_label in samples:
                info = {'img_prefix': self.data_prefix}
                info['img_info'] = {'filename': filename}
                info['gt_label'] = np.array(gt_label, dtype=np.int64)
                data_infos.append(info)
            return data_infos


    def evaluate(self,
                 results,
                 metric='mAP',
                 metric_options=None,
                 logger=None,
                 max_rank=20):
        eval_results = dict()
        if isinstance(metric, list):
            metrics = metric
        elif isinstance(metric, str):
            metrics = [metric]
        else:
            raise TypeError('metric must be a list or a str.')
        allowed_metrics = ['mAP', 'CMC']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported.')

        # distance
        features = torch.stack(results).data.cpu()
        n, c = features.size()
        mat = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(n, n)
        distmat = mat + mat.t()
        distmat.addmm_(features, features.t(), beta=1, alpha=-2)
        distmat = distmat.numpy()

        pids = np.asarray([data_info['gt_label'] for data_info in self.data_infos])
        indices = np.argsort(distmat, axis=1)
        # matches 第i个query相似的第j张图是否id相同
        matches = (pids[indices] == pids[:, np.newaxis]).astype(np.int32)

        all_cmc = []
        all_AP = []
        num_valid_q = 0.
        for q_idx in range(len(results)):
            # compute cmc curve remove self
            raw_cmc = matches[q_idx][1:]  # binary vector, positions with value 1 are correct matches
            if not np.any(raw_cmc):
                # this condition is true when query identity does not appear in gallery
                continue

            cmc = raw_cmc.cumsum()
            cmc[cmc > 1] = 1

            all_cmc.append(cmc[:max_rank])
            num_valid_q += 1.

            # compute average precision
            # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
            num_rel = raw_cmc.sum()
            tmp_cmc = raw_cmc.cumsum()
            tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
            AP = tmp_cmc.sum() / num_rel
            all_AP.append(AP)

        assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

        all_cmc = np.asarray(all_cmc).astype(np.float32)
        all_cmc = all_cmc.sum(0) / num_valid_q
        mAP = np.mean(all_AP)

        eval_results = {'mAP':float(f'{(mAP):.3f}')}
        return eval_results