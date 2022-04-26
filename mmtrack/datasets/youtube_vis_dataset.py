# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
import tempfile
import zipfile
from collections import defaultdict

import mmcv
import numpy as np
from mmcv.utils import print_log
from mmdet.datasets import DATASETS

from mmtrack.core import eval_vis, results2outs
from .coco_video_dataset import CocoVideoDataset


@DATASETS.register_module()
class YouTubeVISDataset(CocoVideoDataset):
    """YouTube VIS dataset for video instance segmentation."""

    CLASSES_2019_version = ('person', 'giant_panda', 'lizard', 'parrot',
                            'skateboard', 'sedan', 'ape', 'dog', 'snake',
                            'monkey', 'hand', 'rabbit', 'duck', 'cat', 'cow',
                            'fish', 'train', 'horse', 'turtle', 'bear',
                            'motorbike', 'giraffe', 'leopard', 'fox', 'deer',
                            'owl', 'surfboard', 'airplane', 'truck', 'zebra',
                            'tiger', 'elephant', 'snowboard', 'boat', 'shark',
                            'mouse', 'frog', 'eagle', 'earless_seal',
                            'tennis_racket')

    CLASSES_2021_version = ('airplane', 'bear', 'bird', 'boat', 'car', 'cat',
                            'cow', 'deer', 'dog', 'duck', 'earless_seal',
                            'elephant', 'fish', 'flying_disc', 'fox', 'frog',
                            'giant_panda', 'giraffe', 'horse', 'leopard',
                            'lizard', 'monkey', 'motorbike', 'mouse', 'parrot',
                            'person', 'rabbit', 'shark', 'skateboard', 'snake',
                            'snowboard', 'squirrel', 'surfboard',
                            'tennis_racket', 'tiger', 'train', 'truck',
                            'turtle', 'whale', 'zebra')

    def __init__(self, dataset_version, *args, **kwargs):
        self.set_dataset_classes(dataset_version)
        super().__init__(*args, **kwargs)

    @classmethod
    def set_dataset_classes(cls, dataset_version):
        if dataset_version == '2019':
            cls.CLASSES = cls.CLASSES_2019_version
        elif dataset_version == '2021':
            cls.CLASSES = cls.CLASSES_2021_version
        else:
            raise NotImplementedError('Not supported YouTubeVIS dataset'
                                      f'version: {dataset_version}')

    def format_results(self,
                       results,
                       resfile_path=None,
                       metrics=['track_segm'],
                       save_as_json=True):
        """Format the results to a zip file (standard format for YouTube-VIS
        Challenge).

        Args:
            results (dict(list[ndarray])): Testing results of the dataset.
            resfile_path (str, optional): Path to save the formatted results.
                Defaults to None.
            metrics (list[str], optional): The results of the specific metrics
                will be formatted. Defaults to ['track_segm'].
            save_as_json (bool, optional): Whether to save the
                json results file. Defaults to True.

        Returns:
            tuple: (resfiles, tmp_dir), resfiles is the path of the result
            json file, tmp_dir is the temporal directory created for saving
            files.
        """
        assert isinstance(results, dict), 'results must be a dict.'
        if isinstance(metrics, str):
            metrics = [metrics]
        assert 'track_segm' in metrics
        if resfile_path is None:
            tmp_dir = tempfile.TemporaryDirectory()
            resfile_path = tmp_dir.name
        else:
            tmp_dir = None
        resfiles = osp.join(resfile_path, 'results.json')

        inds = [i for i, _ in enumerate(self.data_infos) if _['frame_id'] == 0]
        num_vids = len(inds)
        assert num_vids == len(self.vid_ids)
        inds.append(len(self.data_infos))
        vid_infos = self.coco.load_vids(self.vid_ids)

        json_results = []
        for i in range(num_vids):
            video_id = vid_infos[i]['id']
            # collect data for each instances in a video.
            collect_data = dict()
            for frame_id, (bbox_res, mask_res) in enumerate(
                    zip(results['track_bboxes'][inds[i]:inds[i + 1]],
                        results['track_masks'][inds[i]:inds[i + 1]])):
                outs_track = results2outs(bbox_results=bbox_res)
                bboxes = outs_track['bboxes']
                labels = outs_track['labels']
                ids = outs_track['ids']
                masks = mmcv.concat_list(mask_res)
                assert len(masks) == len(bboxes)
                for j, id in enumerate(ids):
                    if id not in collect_data:
                        collect_data[id] = dict(
                            category_ids=[], scores=[], segmentations=dict())
                    collect_data[id]['category_ids'].append(labels[j])
                    collect_data[id]['scores'].append(bboxes[j][4])
                    if isinstance(masks[j]['counts'], bytes):
                        masks[j]['counts'] = masks[j]['counts'].decode()
                    collect_data[id]['segmentations'][frame_id] = masks[j]

            # transform the collected data into official format
            for id, id_data in collect_data.items():
                output = dict()
                output['video_id'] = video_id
                output['score'] = np.array(id_data['scores']).mean().item()
                # majority voting for sequence category
                output['category_id'] = np.bincount(
                    np.array(id_data['category_ids'])).argmax().item() + 1
                output['segmentations'] = []
                for frame_id in range(inds[i + 1] - inds[i]):
                    if frame_id in id_data['segmentations']:
                        output['segmentations'].append(
                            id_data['segmentations'][frame_id])
                    else:
                        output['segmentations'].append(None)
                json_results.append(output)

        if not save_as_json:
            return json_results
        mmcv.dump(json_results, resfiles)

        # zip the json file in order to submit to the test server.
        zip_file_name = osp.join(resfile_path, 'submission_file.zip')
        zf = zipfile.ZipFile(zip_file_name, 'w', zipfile.ZIP_DEFLATED)
        print_log(f"zip the 'results.json' into '{zip_file_name}', "
                  'please submmit the zip file to the test server')
        zf.write(resfiles, 'results.json')
        zf.close()

        return resfiles, tmp_dir

    def evaluate(self, results, metric=['track_segm'], logger=None):
        """Evaluation in COCO protocol.

        Args:
            results (dict): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'track_segm'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """
        if isinstance(metric, list):
            metrics = metric
        elif isinstance(metric, str):
            metrics = [metric]
        else:
            raise TypeError('metric must be a list or a str.')
        allowed_metrics = ['track_segm']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported.')

        eval_results = dict()
        test_results = self.format_results(results, save_as_json=False)
        vis_results = self.convert_back_to_vis_format()
        track_segm_results = eval_vis(test_results, vis_results, logger)
        eval_results.update(track_segm_results)

        return eval_results

    def convert_back_to_vis_format(self):
        """Convert the annotation back to the format of YouTube-VIS. The main
        difference between the two is the format of 'annotation'. Before
        modification, it is recorded in the unit of images, and after
        modification, it is recorded in the unit of instances.This operation is
        to make it easier to use the official eval API.

        Returns:
            dict: A dict with 3 keys, ``categories``, ``annotations``
                and ``videos``.
            - | ``categories`` (list[dict]): Each dict has 2 keys,
                ``id`` and ``name``.
            - | ``videos`` (list[dict]): Each dict has 4 keys of video info,
                ``id``, ``name``, ``width`` and ``height``.
            - | ``annotations`` (list[dict]): Each dict has 7 keys of video
                info, ``category_id``, ``segmentations``, ``bboxes``,
                ``video_id``, ``areas``, ``id`` and ``iscrowd``.
        """

        vis_anns = defaultdict(list)

        vis_anns['categories'] = copy.deepcopy(self.coco.dataset['categories'])
        vis_anns['videos'] = copy.deepcopy(self.coco.dataset['videos'])

        len_videos = dict()  # mapping from video_id to video_length
        for video_id, video_infos in self.coco.vidToImgs.items():
            len_videos[video_id] = len(video_infos)

        for video_id, ins_ids in self.coco.vidToInstances.items():
            cur_video_len = len_videos[video_id]
            for ins_id in ins_ids:
                # In the official format, no instances are represented by
                # 'None', however, only images with instances are recorded
                # in the current annotations, so we need to use 'None' to
                # initialize these lists.
                segm = [None] * cur_video_len
                bbox = [None] * cur_video_len
                area = [None] * cur_video_len
                category_id = None
                iscrowd = None
                for img_id in self.coco.instancesToImgs.get(ins_id):
                    frame_id = self.coco.imgs[img_id]['frame_id']
                    for ann in self.coco.imgToAnns[img_id]:
                        if ann['instance_id'] == ins_id:
                            segm[frame_id] = ann['segmentation']
                            bbox[frame_id] = ann['bbox']
                            area[frame_id] = ann['area']
                            category_id = ann['category_id']
                            iscrowd = ann['iscrowd']
                assert category_id is not None
                instance = dict(
                    category_id=category_id,
                    segmentations=segm,
                    bboxes=bbox,
                    video_id=video_id,
                    areas=area,
                    id=ins_id,
                    iscrowd=iscrowd)
                vis_anns['annotations'].append(instance)

        return dict(vis_anns)
