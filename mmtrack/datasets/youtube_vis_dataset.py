# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
import zipfile

import mmcv
import numpy as np
from mmcv.utils import print_log
from mmdet.datasets import DATASETS

from mmtrack.core import results2outs
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
                       metrics=['track_segm']):
        """Format the results to a zip file (standard format for YouTube-VIS
        Challenge).

        Args:
            results (dict(list[ndarray])): Testing results of the dataset.
            resfile_path (str, optional): Path to save the formatted results.
                Defaults to None.
            metrics (list[str], optional): The results of the specific metrics
                will be formatted. Defaults to ['track_segm'].

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
                for i, id in enumerate(ids):
                    if id not in collect_data:
                        collect_data[id] = dict(
                            category_ids=[], scores=[], segmentations=dict())
                    collect_data[id]['category_ids'].append(labels[i])
                    collect_data[id]['scores'].append(bboxes[i][4])
                    if isinstance(masks[i]['counts'], bytes):
                        masks[i]['counts'] = masks[i]['counts'].decode()
                    collect_data[id]['segmentations'][frame_id] = masks[i]

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
        mmcv.dump(json_results, resfiles)

        # zip the json file in order to submit to the test server.
        zip_file_name = osp.join(resfile_path, 'submission_file.zip')
        zf = zipfile.ZipFile(zip_file_name, 'w', zipfile.ZIP_DEFLATED)
        print_log(f"zip the 'results.json' into '{zip_file_name}', "
                  'please submmit the zip file to the test server')
        zf.write(resfiles, 'results.json')
        zf.close()

        return resfiles, tmp_dir
