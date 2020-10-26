import os
import tempfile

import mmcv
from mmcv.utils import print_log
from mmdet.datasets import DATASETS
from pycocotools.coco import COCO

from .coco_video_dataset import CocoVideoDataset
from .parsers import CocoVID


@DATASETS.register_module()
class TaoDataset(CocoVideoDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_annotations(self, ann_file):
        """Load annotation from annotation file."""
        if not self.load_as_video:
            data_infos = self.load_lvis_anns(ann_file)
        else:
            data_infos = self.load_tao_anns(ann_file)
        return data_infos

    def load_lvis_anns(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            if info['file_name'].startswith('COCO'):
                # Convert form the COCO 2014 file naming convention of
                # COCO_[train/val/test]2014_000000000000.jpg to the 2017
                # naming convention of 000000000000.jpg
                # (LVIS v1 will fix this naming issue)
                info['filename'] = info['file_name'][-16:]
            else:
                info['filename'] = info['file_name']
            data_infos.append(info)
        return data_infos

    def load_tao_anns(self, ann_file):
        self.coco = CocoVID(ann_file)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}

        data_infos = []
        self.vid_ids = self.coco.get_vid_ids()
        self.img_ids = []
        for vid_id in self.vid_ids:
            img_ids = self.coco.get_img_ids_from_vid(vid_id)
            if self.key_img_sampler is not None:
                img_ids = self.key_img_sampling(img_ids,
                                                **self.key_img_sampler)
            self.img_ids.extend(img_ids)
            for img_id in img_ids:
                info = self.coco.load_imgs([img_id])[0]
                if info['file_name'].startswith('COCO'):
                    # Convert form the COCO 2014 file naming convention of
                    # COCO_[train/val/test]2014_000000000000.jpg to the 2017
                    # naming convention of 000000000000.jpg
                    # (LVIS v1 will fix this naming issue)
                    info['filename'] = info['file_name'][-16:]
                else:
                    info['filename'] = info['file_name']
                data_infos.append(info)
        return data_infos

    def _track2json(self, results):
        """Convert tracking results to TAO json style."""
        inds = [i for i, _ in enumerate(self.data_infos) if _['frame_id'] == 0]
        num_vids = len(inds)
        inds.append(len(self.data_infos))
        results = [results[inds[i]:inds[i + 1]] for i in range(num_vids)]
        img_infos = [
            self.data_infos[inds[i]:inds[i + 1]] for i in range(num_vids)
        ]

        json_results = []
        max_track_id = 0
        for _img_infos, _results in zip(img_infos, results):
            track_ids = []
            for img_info, result in zip(_img_infos, _results):
                img_id = img_info['id']
                for label in range(len(result)):
                    bboxes = result[label]
                    for i in range(bboxes.shape[0]):
                        data = dict()
                        data['image_id'] = img_id
                        data['bbox'] = self.xyxy2xywh(bboxes[i, 1:])
                        data['score'] = float(bboxes[i][-1])
                        data['category_id'] = self.cat_ids[label]
                        data['video_id'] = img_info['video_id']
                        data['track_id'] = max_track_id + int(bboxes[i][0])
                        track_ids.append(int(bboxes[i][0]))
                        json_results.append(data)
            track_ids = list(set(track_ids))
            max_track_id += max(track_ids) + 1

        return json_results

    def format_results(self, results, resfile_path=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[tuple | numpy.ndarray]): Testing results of the
                dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing \
                the json filepaths, tmp_dir is the temporal directory created \
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, dict), 'results must be a list'
        assert 'track_results' in results
        assert 'bbox_results' in results

        if resfile_path is None:
            tmp_dir = tempfile.TemporaryDirectory()
            resfile_path = tmp_dir.name
        else:
            tmp_dir = None
        os.makedirs(resfile_path, exist_ok=True)
        result_files = dict()

        bbox_results = self._det2json(results['bbox_results'])
        result_files['bbox'] = f'{resfile_path}/tao_bbox.json'
        mmcv.dump(bbox_results, result_files['bbox'])

        track_results = self._track2json(results['track_results'])
        result_files['track'] = f'{resfile_path}/tao_track.json'
        mmcv.dump(track_results, result_files['track'])

        return result_files, tmp_dir

    def evaluate(self,
                 results,
                 metric=['track'],
                 logger=None,
                 resfile_path=None):
        if isinstance(metric, list):
            metrics = metric
        elif isinstance(metric, str):
            metrics = [metric]
        else:
            raise TypeError('metric must be a list or a str.')
        allowed_metrics = ['bbox', 'track']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported.')

        result_files, tmp_dir = self.format_results(results, resfile_path)

        eval_results = dict()

        if 'track' in metrics:
            from tao.toolkit.tao import TaoEval
            print_log('Evaluating TAO results...', logger)
            tao_eval = TaoEval(self.ann_file, result_files['track'])
            tao_eval.params.img_ids = self.img_ids
            tao_eval.params.cat_ids = self.cat_ids
            tao_eval.run()

            tao_eval.print_results()
            tao_results = tao_eval.get_results()
            for k, v in tao_results.items():
                if isinstance(k, str) and k.startswith('AP'):
                    key = 'track_{}'.format(k)
                    val = float('{:.3f}'.format(float(v)))
                    eval_results[key] = val

        if 'bbox' in metrics:
            try:
                import lvis
                assert lvis.__version__ >= '10.5.3'
                from lvis import LVIS, LVISResults, LVISEval
            except AssertionError:
                raise AssertionError(
                    'Incompatible version of lvis is installed. '
                    'Run pip uninstall lvis first. Then run pip '
                    'install mmlvis to install open-mmlab forked '
                    'lvis. ')
            except ImportError:
                raise ImportError(
                    'Package lvis is not installed. Please run pip '
                    'install mmlvis to install open-mmlab forked '
                    'lvis.')
            print_log('Evaluating detection results...', logger)
            lvis_gt = LVIS(self.ann_file)
            lvis_dt = LVISResults(lvis_gt, result_files['bbox'])
            lvis_eval = LVISEval(lvis_gt, lvis_dt, 'bbox')
            lvis_eval.params.imgIds = self.img_ids
            lvis_eval.params.catIds = self.cat_ids
            lvis_eval.evaluate()
            lvis_eval.accumulate()
            lvis_eval.summarize()
            lvis_eval.print_results()
            lvis_results = lvis_eval.get_results()
            for k, v in lvis_results.items():
                if k.startswith('AP'):
                    key = '{}_{}'.format('bbox', k)
                    val = float('{:.3f}'.format(float(v)))
                    eval_results[key] = val
            ap_summary = ' '.join([
                '{}:{:.3f}'.format(k, float(v))
                for k, v in lvis_results.items() if k.startswith('AP')
            ])
            eval_results['bbox_mAP_copypaste'] = ap_summary

        if tmp_dir is not None:
            tmp_dir.cleanup()

        return eval_results
