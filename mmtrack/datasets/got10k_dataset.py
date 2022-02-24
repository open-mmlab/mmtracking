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
class GOT10kDataset(BaseSOTDataset):
    """GOT10k Dataset of single object tracking.

    The dataset can both support training and testing mode.
    """

    def __init__(self, *args, **kwargs):
        super(GOT10kDataset, self).__init__(*args, **kwargs)

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
                        'start_frame_id': the starting frame number contained
                            in the image name
                        'end_frame_id': the ending frame number contained in
                            the image name
                        'framename_template': the template of image name
                    }
        """
        print('Loading GOT10k dataset...')
        start_time = time.time()
        assert split in ['train', 'val', 'test', 'val_vot', 'train_vot']
        data_infos = []
        if split in ['train', 'val', 'test']:
            videos_list = np.loadtxt(
                osp.join(self.img_prefix, split, 'list.txt'), dtype=np.str_)
        else:
            split = '_'.join(split.split('_')[::-1])
            vids_id_list = np.loadtxt(
                osp.join(self.img_prefix, 'train',
                         f'got10k_{split}_split.txt'),
                dtype=float)
            videos_list = [
                'GOT-10k_Train_%06d' % (int(video_id) + 1)
                for video_id in vids_id_list
            ]

        videos_list = sorted(videos_list)
        for video_name in videos_list:
            if split in ['val', 'test']:
                video_path = osp.join(split, video_name)
            else:
                video_path = osp.join('train', video_name)
            ann_path = osp.join(video_path, 'groundtruth.txt')
            img_names = glob.glob(
                osp.join(self.img_prefix, video_path, '*.jpg'))
            end_frame_name = max(
                img_names, key=lambda x: int(osp.basename(x).split('.')[0]))
            end_frame_id = int(osp.basename(end_frame_name).split('.')[0])
            data_infos.append(
                dict(
                    video_path=video_path,
                    ann_path=ann_path,
                    start_frame_id=1,
                    end_frame_id=end_frame_id,
                    framename_template='%08d.jpg'))
        print(f'GOT10k dataset loaded! ({time.time()-start_time:.2f} s)')
        return data_infos

    def get_visibility_from_video(self, video_ind):
        """Get the visible information of instance in a video."""
        if not self.test_mode:
            absense_info_path = osp.join(
                self.img_prefix, self.data_infos[video_ind]['video_path'],
                'absence.label')
            cover_info_path = osp.join(
                self.img_prefix, self.data_infos[video_ind]['video_path'],
                'cover.label')
            absense_info = np.loadtxt(absense_info_path, dtype=bool)
            # The values of key 'cover' are
            # int numbers in range [0,8], which correspond to
            # ranges of object visible ratios: 0%, (0%, 15%],
            # (15%~30%], (30%, 45%], (45%, 60%],(60%, 75%],
            # (75%, 90%], (90%, 100%) and 100% respectively
            cover_info = np.loadtxt(cover_info_path, dtype=int)
            visible = ~absense_info & (cover_info > 0)
            visible_ratio = cover_info / 8.
            return dict(visible=visible, visible_ratio=visible_ratio)
        else:
            return super(GOT10kDataset,
                         self).get_visibility_from_video(video_ind)

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
        """Format the results to txts (standard format for GOT10k Challenge).

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

        # transform tracking results format
        # from [bbox_1, bbox_2, ...] to {'video_1':[bbox_1, bbox_2, ...], ...}
        track_bboxes = results['track_bboxes']
        print('-------- There are total {} images --------'.format(
            len(track_bboxes)))

        start_ind = end_ind = 0
        for num, video_info in zip(self.num_frames_per_video, self.data_infos):
            end_ind += num
            video_name = video_info['video_path'].split(os.sep)[-1]
            video_resfiles_path = osp.join(resfile_path, video_name)
            if not osp.isdir(video_resfiles_path):
                os.makedirs(video_resfiles_path, exist_ok=True)
            video_bbox_txt = osp.join(video_resfiles_path,
                                      '{}_001.txt'.format(video_name))
            video_time_txt = osp.join(video_resfiles_path,
                                      '{}_time.txt'.format(video_name))
            with open(video_bbox_txt,
                      'w') as f_bbox, open(video_time_txt, 'w') as f_time:

                for bbox in results['track_bboxes'][start_ind:end_ind]:
                    bbox = [
                        str(f'{bbox[0]:.4f}'),
                        str(f'{bbox[1]:.4f}'),
                        str(f'{(bbox[2] - bbox[0]):.4f}'),
                        str(f'{(bbox[3] - bbox[1]):.4f}')
                    ]
                    line = ','.join(bbox) + '\n'
                    f_bbox.writelines(line)
                    # We don't record testing time, so we set a default
                    # time in order to test on the server.
                    f_time.writelines('0.0001\n')
            start_ind += num

        shutil.make_archive(resfile_path, 'zip', resfile_path)
        shutil.rmtree(resfile_path)
