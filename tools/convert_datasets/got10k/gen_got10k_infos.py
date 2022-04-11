# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import os
import os.path as osp
import time

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate the information of GOT10k dataset')
    parser.add_argument(
        '-i',
        '--input',
        help='root directory of GOT10k dataset',
    )
    parser.add_argument(
        '-o',
        '--output',
        help='directory to save text file',
    )
    parser.add_argument(
        '--split',
        help="the split set of GOT10k, 'all' denotes the whole dataset",
        choices=['train', 'test', 'val', 'train_vot', 'val_vot', 'all'],
        default='all')
    return parser.parse_args()


def gen_data_infos(data_root, save_dir, split='train'):
    """Generate dataset information.

    Args:
        data_root (str): The path of dataset.
        save_dir (str): The path to save the information of dataset.
        split (str): the split ('train' or 'test') of dataset.
    """
    print(f'Generate the information of {split} set of LaSOT dataset...')
    start_time = time.time()
    assert split in ['train', 'val', 'test', 'val_vot', 'train_vot']
    if split in ['train', 'val', 'test']:
        videos_list = np.loadtxt(
            osp.join(data_root, split, 'list.txt'), dtype=np.str_)
    else:
        split_reverse = '_'.join(split.split('_')[::-1])
        vids_id_list = np.loadtxt(
            osp.join(data_root, 'train', f'got10k_{split_reverse}_split.txt'),
            dtype=float)
        videos_list = [
            'GOT-10k_Train_%06d' % (int(video_id) + 1)
            for video_id in vids_id_list
        ]

    if not osp.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    with open(osp.join(save_dir, f'got10k_{split}_infos.txt'), 'w') as f:
        f.write('The format of each line in this txt is '
                '(video_path,annotation_path,start_frame_id,end_frame_id)')
        videos_list = sorted(videos_list)
        for video_name in videos_list:
            if split in ['val', 'test']:
                video_path = osp.join(split, video_name)
            else:
                video_path = osp.join('train', video_name)
            ann_path = osp.join(video_path, 'groundtruth.txt')
            img_names = glob.glob(osp.join(data_root, video_path, '*.jpg'))
            end_frame_name = max(
                img_names, key=lambda x: int(osp.basename(x).split('.')[0]))
            end_frame_id = int(osp.basename(end_frame_name).split('.')[0])
            f.write(f'\n{video_path},{ann_path},1,{end_frame_id}')

    print(f'Done! ({time.time()-start_time:.2f} s)')
    print(f'The results are saved in {save_dir}')


def main():
    args = parse_args()
    if args.split == 'all':
        for split in ['train', 'test', 'val', 'train_vot', 'val_vot']:
            gen_data_infos(args.input, args.output, split=split)
    else:
        gen_data_infos(args.input, args.output, split=args.split)


if __name__ == '__main__':
    main()
