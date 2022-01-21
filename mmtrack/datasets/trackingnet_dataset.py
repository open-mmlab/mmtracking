# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os
import os.path as osp
import shutil
import time

from mmdet.datasets import DATASETS

from .base_sot_dataset import BaseSOTDataset


@DATASETS.register_module()
class TrackingNetDataset(BaseSOTDataset):
    """TrackingNet dataset of single object tracking.

    The dataset can both support training and testing mode.
    """

    def __init__(self, num_chunks=12, *args, **kwargs):
        """Initialization of SOT dataset class.

        Args:
            num_chunks (int, optional): the number of chunks. Some methods may
                only use part of the dataset. Default to all chunks, 12.
        """
        self.num_chunks = num_chunks
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
            chunks = [f'TRAIN_{i}' for i in range(self.num_chunks)]
        else:
            raise NotImplementedError

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
            video_name = video_info['video_path'].split('/')[-1]
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
