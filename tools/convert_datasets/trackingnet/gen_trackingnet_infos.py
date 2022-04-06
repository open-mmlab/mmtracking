# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import os
import os.path as osp
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate the information of TrackingNet dataset')
    parser.add_argument(
        '-i',
        '--input',
        help='root directory of TrackingNet dataset',
    )
    parser.add_argument(
        '-o',
        '--output',
        help='directory to save text file',
    )
    parser.add_argument(
        '--split',
        help="the split set of TrackingNet,'all' denotes the whole dataset",
        choices=['train', 'test', 'all'],
        default='all')
    parser.add_argument(
        '--chunks',
        help='the chunks of train set of TrackingNet',
        nargs='+',
        type=int,
        default=['all'])
    return parser.parse_args()


def gen_data_infos(data_root, save_dir, split='train', chunks=['all']):
    """Generate dataset information.

    args:
        data_root (str): The path of dataset.
        save_dir (str): The path to save the information of dataset.
        split (str): the split ('train' or 'test') of dataset.
        chunks (list): the chunks of train set of TrackingNet.
    """
    print(f'Generate the information of {split} set of TrackingNet dataset...')
    start_time = time.time()
    if split == 'test':
        chunks_list = ['TEST']
    elif split == 'train':
        if 'all' in chunks:
            chunks_list = [f'TRAIN_{i}' for i in range(12)]
        else:
            chunks_list = [f'TRAIN_{chunk}' for chunk in chunks]
    else:
        raise NotImplementedError

    if not osp.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    assert len(chunks_list) > 0
    with open(osp.join(save_dir, f'trackingnet_{split}_infos.txt'), 'w') as f:
        f.write(
            'The format of each line in this txt is '
            '(chunk,video_path,annotation_path,start_frame_id,end_frame_id)')
        for chunk in chunks_list:
            chunk_ann_dir = osp.join(data_root, chunk)
            assert osp.isdir(
                chunk_ann_dir
            ), f'annotation directory {chunk_ann_dir} does not exist'

            videos_list = sorted(os.listdir(osp.join(chunk_ann_dir, 'frames')))
            for video_name in videos_list:
                video_path = osp.join(chunk, 'frames', video_name)
                # avoid creating empty file folds by mistakes
                if not os.listdir(osp.join(data_root, video_path)):
                    continue
                ann_path = osp.join(chunk, 'anno', video_name + '.txt')
                img_names = glob.glob(osp.join(data_root, video_path, '*.jpg'))
                end_frame_name = max(
                    img_names,
                    key=lambda x: int(osp.basename(x).split('.')[0]))
                end_frame_id = int(osp.basename(end_frame_name).split('.')[0])
                f.write(f'\n{video_path},{ann_path},0,{end_frame_id}')

    print(f'Done! ({time.time()-start_time:.2f} s)')
    print(f'The results are saved in {save_dir}')


def main():
    args = parse_args()
    assert set(args.chunks).issubset(set(range(12)) | {'all'})
    if args.split == 'all':
        for split in ['train', 'test']:
            gen_data_infos(
                args.input, args.output, split=split, chunks=args.chunks)
    else:
        gen_data_infos(
            args.input, args.output, split=args.split, chunks=args.chunks)


if __name__ == '__main__':
    main()
