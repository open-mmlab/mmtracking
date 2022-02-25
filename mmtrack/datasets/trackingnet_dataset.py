# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os
import os.path as osp
import shutil
import time

import numpy as np
from mmdet.datasets import DATASETS

from .base_sot_dataset import BaseSOTDataset


@DATASETS.register_module()
class TrackingNetDataset(BaseSOTDataset):
    """TrackingNet dataset of single object tracking.

    The dataset can both support training and testing mode.
    """

    def __init__(self, chunks_list=['all'], *args, **kwargs):
        """Initialization of SOT dataset class.

        Args:
            chunks_list (list, optional): the training chunks. The optional
                values in this list are: 0, 1, 2, ..., 10, 11 and 'all'. Some
                methods may only use part of the dataset. Default to all
                chunks, namely ['all'].
        """
        if isinstance(chunks_list, (str, int)):
            chunks_list = [chunks_list]
        assert set(chunks_list).issubset(set(range(12)) | {'all'})
        self.chunks_list = chunks_list
        super(TrackingNetDataset, self).__init__(*args, **kwargs)

    def load_data_infos(self, split='train'):
        """Load dataset information.

        Args:
            split (str, optional): the split of dataset. Defaults to 'train'.

        Returns:
            list[dict]: the length of the list is the number of videos. The
                inner dict is in the following format:
                    {
                        'video_path': the video path
                        'ann_path': the annotation path
                        'start_frame_id': the starting frame ID number
                            contained in the image name
                        'end_frame_id': the ending frame ID number contained in
                            the image name
                        'framename_template': the template of image name
                    }
        """
        print('Loading TrackingNet dataset...')
        start_time = time.time()
        if split == 'test':
            chunks = ['TEST']
        elif split == 'train':
            if 'all' in self.chunks_list:
                chunks = [f'TRAIN_{i}' for i in range(12)]
            else:
                chunks = [f'TRAIN_{chunk}' for chunk in self.chunks_list]
        else:
            raise NotImplementedError

        assert len(chunks) > 0
        data_infos = []
        for chunk in chunks:
            chunk_ann_dir = osp.join(self.img_prefix, chunk)
            assert osp.isdir(
                chunk_ann_dir
            ), f'annotation directory {chunk_ann_dir} does not exist'

            videos_list = sorted(os.listdir(osp.join(chunk_ann_dir, 'frames')))
            for video_name in videos_list:
                video_path = osp.join(chunk, 'frames', video_name)
                # avoid creating empty file folds by mistakes
                if not os.listdir(osp.join(self.img_prefix, video_path)):
                    continue
                ann_path = osp.join(chunk, 'anno', video_name + '.txt')
                img_names = glob.glob(
                    osp.join(self.img_prefix, video_path, '*.jpg'))
                end_frame_name = max(
                    img_names,
                    key=lambda x: int(osp.basename(x).split('.')[0]))
                end_frame_id = int(osp.basename(end_frame_name).split('.')[0])
                data_info = dict(
                    video_path=video_path,
                    ann_path=ann_path,
                    start_frame_id=0,
                    end_frame_id=end_frame_id,
                    framename_template='%d.jpg')
                data_infos.append(data_info)
        print(f'TrackingNet dataset loaded! ({time.time()-start_time:.2f} s)')
        return data_infos

    def prepare_test_data(self, video_ind, frame_ind):
        """Get testing data of one frame. We parse one video, get one frame
        from it and pass the frame information to the pipeline.

        Args:
            video_ind (int): video index
            frame_ind (int): frame index

        Returns:
            dict: testing data of one frame.
        """
        if self.test_memo.get('video_ind', None) != video_ind:
            self.test_memo.video_ind = video_ind
            self.test_memo.img_infos = self.get_img_infos_from_video(video_ind)
        assert 'video_ind' in self.test_memo and 'img_infos' in self.test_memo

        img_info = dict(
            filename=self.test_memo.img_infos['filename'][frame_ind],
            frame_id=frame_ind)
        if frame_ind == 0:
            ann_infos = self.get_ann_infos_from_video(video_ind)
            ann_info = dict(
                bboxes=ann_infos['bboxes'][frame_ind], visible=True)
        else:
            ann_info = dict(
                bboxes=np.array([0] * 4, dtype=np.float32), visible=True)

        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        results = self.pipeline(results)
        return results

    def format_results(self, results, resfile_path=None):
        """Format the results to txts (standard format for TrackingNet
        Challenge).

        Args:
            results (dict(list[ndarray])): Testing results of the dataset.
            resfile_path (str): Path to save the formatted results.
                Defaults to None.
        """
        # prepare saved dir
        assert resfile_path is not None, 'Please give key-value pair \
            like resfile_path=xxx in argparse'

        if not osp.isdir(resfile_path):
            os.makedirs(resfile_path, exist_ok=True)

        print('-------- There are total {} images --------'.format(
            len(results['track_bboxes'])))

        # transform tracking results format
        # from [bbox_1, bbox_2, ...] to {'video_1':[bbox_1, bbox_2, ...], ...}
        start_ind = end_ind = 0
        for num, video_info in zip(self.num_frames_per_video, self.data_infos):
            end_ind += num
            video_name = video_info['video_path'].split(os.sep)[-1]
            video_txt = osp.join(resfile_path, '{}.txt'.format(video_name))
            with open(video_txt, 'w') as f:
                for bbox in results['track_bboxes'][start_ind:end_ind]:
                    bbox = [
                        str(f'{bbox[0]:.4f}'),
                        str(f'{bbox[1]:.4f}'),
                        str(f'{(bbox[2] - bbox[0]):.4f}'),
                        str(f'{(bbox[3] - bbox[1]):.4f}')
                    ]
                    line = ','.join(bbox) + '\n'
                    f.writelines(line)
            start_ind += num

        shutil.make_archive(resfile_path, 'zip', resfile_path)
        shutil.rmtree(resfile_path)
