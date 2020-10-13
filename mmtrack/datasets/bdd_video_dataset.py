import os
import os.path as osp
import tempfile

import mmcv
from mmcv.utils import print_log
from mmdet.datasets import DATASETS

from .coco_video_dataset import CocoVideoDataset


@DATASETS.register_module()
class BDDVideoDataset(CocoVideoDataset):

    CLASSES = ('pedestrian', 'rider', 'car', 'bus', 'truck', 'bicycle',
               'motorcycle', 'train')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # TODO: adapt
    def format_bdd100k_results(self, results, resfile_path=None):
        assert isinstance(results, dict)
        assert 'bbox_results' in results
        assert 'track_results' in results
        if resfile_path is None:
            raise NotImplementedError('waiting for offical API.')
        for task in ['track', 'bbox']:
            self.format_bdd100k(
                results[f'{task}_results'], resfile_path, task=task)

    def _format_bdd100k(self, results, resfile_path=None, task='track'):
        print_log(f'formatting {task} results for bdd100k.', self.logger)
        assert task in ['track', 'bbox']
        if resfile_path is None:
            tmp_dir = tempfile.TemporaryDirectory()
            resfile_path = osp.join(tmp_dir.name, task)
        else:
            tmp_dir = None
            resfile_path = osp.join(resfile_path, task)
            if osp.exists(resfile_path):
                print_log('remove previous results.', self.logger)
                import shutil
                shutil.rmtree(resfile_path)
            os.makedirs(resfile_path, exist_ok=False)
        inds = [i for i, _ in enumerate(self.data_infos) if _['frame_id'] == 0]
        num_vids = len(inds)
        assert num_vids == len(self.vid_ids)
        inds.append(len(self.data_infos))
        vid_infos = self.coco.load_vids(self.vid_ids)
        vid_names = [_['name'] for _ in vid_infos]
        for i in range(num_vids):
            resinfos = []
            resfile = osp.join(resfile_path, f'{vid_names[i]}.json')
            result = results[inds[i]:inds[i + 1]]
            data_info = self.data_infos[inds[i]:inds[i + 1]]
            assert len(result) == len(data_info)
            for info, res in zip(data_info, result):
                resinfo = dict()
                resinfo['video_name'] = vid_names[i]
                resinfo['name'] = info['file_name']
                resinfo['index'] = info['frame_id']
                resinfo['labels'] = []
                for cls_id, bboxes in enumerate(res):
                    category = self.CLASSES[cls_id]
                    for bbox in bboxes:
                        resbbox = dict(category=category)
                        if len(bbox) == 6:
                            id, x1, y1, x2, y2, score = map(float, bbox)
                            resbbox['id'] = int(id)
                        elif len(bbox) == 5:
                            x1, y1, x2, y2, score = map(float, bbox)
                        else:
                            raise ValueError('length of bbox must be 5 or 6.')
                        resbbox['box2d'] = dict(x1=x1, y1=y1, x2=x2, y2=y2)
                        resbbox['score'] = score
                        resinfo['labels'].append(resbbox)
                resinfos.append(resinfo)
            mmcv.dump(resinfos, resfile)
