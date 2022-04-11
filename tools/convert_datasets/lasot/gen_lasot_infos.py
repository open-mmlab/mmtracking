# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import os
import os.path as osp
import time

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate the information of LaSOT dataset')
    parser.add_argument(
        '-i',
        '--input',
        help='root directory of LaSOT dataset',
    )
    parser.add_argument(
        '-o',
        '--output',
        help='directory to save text file',
    )
    parser.add_argument(
        '--split',
        help="the split set of LaSOT, 'all' denotes the whole dataset",
        choices=['train', 'test', 'all'],
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
    assert split in ['train', 'test']

    test_videos_list = np.loadtxt(
        osp.join(osp.dirname(__file__), 'testing_set.txt'), dtype=np.str_)
    if split == 'test':
        videos_list = test_videos_list.tolist()
    else:
        all_videos_list = glob.glob(data_root + '/*/*-*')
        test_videos = set(test_videos_list)
        videos_list = []
        for x in all_videos_list:
            x = osp.basename(x)
            if x not in test_videos:
                videos_list.append(x)

    if not osp.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    videos_list = sorted(videos_list)
    with open(osp.join(save_dir, f'lasot_{split}_infos.txt'), 'w') as f:
        f.write('The format of each line in this txt is '
                '(video_path,annotation_path,start_frame_id,end_frame_id)')
        for video_name in videos_list:
            video_name = osp.join(video_name.split('-')[0], video_name)
            video_path = osp.join(video_name, 'img')
            ann_path = osp.join(video_name, 'groundtruth.txt')
            img_names = glob.glob(
                osp.join(data_root, video_name, 'img', '*.jpg'))
            end_frame_name = max(
                img_names, key=lambda x: int(osp.basename(x).split('.')[0]))
            end_frame_id = int(osp.basename(end_frame_name).split('.')[0])
            f.write(f'\n{video_path},{ann_path},1,{end_frame_id}')

    print(f'Done! ({time.time()-start_time:.2f} s)')
    print(f'The results are saved in {save_dir}')


def main():
    args = parse_args()
    if args.split == 'all':
        for split in ['train', 'test']:
            gen_data_infos(args.input, args.output, split=split)
    else:
        gen_data_infos(args.input, args.output, split=args.split)


if __name__ == '__main__':
    main()
