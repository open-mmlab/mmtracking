import copy
from collections import defaultdict

import numpy as np
import torch
from mmcls.datasets import DATASETS, BaseDataset


@DATASETS.register_module()
class ReIDDataset(BaseDataset):

    def __init__(self,
                 load_as_video=False,
                 load_as_reid=True,
                 triplet_sampler=dict(num_ids=1, ins_per_id=1),
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        assert not load_as_video, \
            'reid dataset can not be loaded as COCO style.'

        self.load_as_reid = load_as_reid
        self.load_as_video = load_as_video
        self.triplet_sampler = triplet_sampler
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
        self.parse_annotations(data_infos)
        return data_infos

    def parse_annotations(self, data_infos):
        index_tmp_dic = defaultdict(list)
        self.index_dic = dict()
        for idx, info in enumerate(data_infos):
            pid = info['gt_label']
            index_tmp_dic[int(pid)].append(idx)
        for pid, idxs in index_tmp_dic.items():
            self.index_dic[pid] = np.asarray(idxs, dtype=np.int64)

        self.pids = np.asarray(list(self.index_dic.keys()), dtype=np.int64)

    def triplet_sampling(self, img_info, num_ids=8, ins_per_id=4):
        assert len(self.pids) >= num_ids, \
            'The number of pedestrian ids in the training set must ' \
            'be greater than the number of pedestrian ids in the sample.'

        pos_pid = img_info['gt_label']
        pos_idxs = self.index_dic[int(pos_pid)]
        idxs_list = []
        # select positive samplers
        idxs_list.extend(pos_idxs[np.random.choice(
            pos_idxs.shape[0], ins_per_id, replace=True)])

        # select negative ids
        neg_pids = np.random.choice(
            [i for i, _ in enumerate(self.pids) if i != pos_pid],
            num_ids - 1,
            replace=False)

        # select negative samplers for each negative id
        for neg_pid in neg_pids:
            neg_idxs = self.index_dic[neg_pid]
            idxs_list.extend(neg_idxs[np.random.choice(
                neg_idxs.shape[0], ins_per_id, replace=True)])

        triplet_img_infos = []
        for idx in idxs_list:
            triplet_img_infos.append(copy.deepcopy(self.data_infos[idx]))

        return triplet_img_infos

    def prepare_data(self, idx):
        img_info = self.data_infos[idx]
        if self.triplet_sampler is not None:
            img_infos = self.triplet_sampling(img_info, **self.triplet_sampler)
            results = copy.deepcopy(img_infos)
        else:
            results = copy.deepcopy(img_info)
        return self.pipeline(results)

    def evaluate(self,
                 results,
                 metric='mAP',
                 metric_options=None,
                 logger=None,
                 max_rank=20):
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
        results = [result.data.cpu() for result in results]
        features = torch.stack(results) if len(
            results[0].size()) == 1 else torch.cat(results)
        n, c = features.size()
        mat = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(n, n)
        distmat = mat + mat.t()
        distmat.addmm_(features, features.t(), beta=1, alpha=-2)
        distmat = distmat.numpy()

        pids = self.get_gt_labels()
        indices = np.argsort(distmat, axis=1)
        matches = (pids[indices] == pids[:, np.newaxis])\
            .astype(np.int32)

        all_cmc = []
        all_AP = []
        num_valid_q = 0.
        for q_idx in range(n):
            # compute cmc curve remove self
            raw_cmc = matches[q_idx][1:]
            if not np.any(raw_cmc):
                # this condition is true when query identity
                # does not appear in gallery
                continue

            cmc = raw_cmc.cumsum()
            cmc[cmc > 1] = 1

            all_cmc.append(cmc[:max_rank])
            num_valid_q += 1.

            # compute average precision
            # reference:
            # https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
            num_rel = raw_cmc.sum()
            tmp_cmc = raw_cmc.cumsum()
            tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
            AP = tmp_cmc.sum() / num_rel
            all_AP.append(AP)

        assert num_valid_q > 0, \
            'Error: all query identities do not appear in gallery'

        all_cmc = np.asarray(all_cmc).astype(np.float32)
        all_cmc = all_cmc.sum(0) / num_valid_q
        mAP = np.mean(all_AP)

        eval_results = {}
        if 'mAP' in metrics:
            eval_results['mAP'] = np.around(mAP, decimals=3)
        if 'CMC' in metrics:
            eval_results['R1'] = np.around(all_cmc[0], decimals=3)
            eval_results['R5'] = np.around(all_cmc[4], decimals=3)
            eval_results['R10'] = np.around(all_cmc[9], decimals=3)
            eval_results['R20'] = np.around(all_cmc[19], decimals=3)

        return eval_results
