# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import os
import os.path as osp
from collections import defaultdict

import mmcv
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='GOT10k dataset to COCO Video format')
    parser.add_argument(
        '-i',
        '--input',
        help='root directory of GOT10k dataset',
    )
    parser.add_argument(
        '-o',
        '--output',
        help='directory to save coco formatted label file',
    )
    parser.add_argument(
        '--split',
        help="the split set of GOT10k, 'all' denotes the whole dataset",
        choices=['train', 'test', 'val', 'all'],
        default='all')
    return parser.parse_args()


def convert_got10k(ann_dir, save_dir, split='test'):
    """Convert got10k dataset to COCO style.

    Args:
        ann_dir (str): The path of got10k dataset
        save_dir (str): The path to save `got10k`.
        split (str): the split ('train'， 'val' or 'test') of dataset.
    """
    assert split in ['train', 'test', 'val'], f'split [{split}] does not exist'
    got10k = defaultdict(list)
    records = dict(vid_id=1, img_id=1, ann_id=1, global_instance_id=1)
    got10k['categories'] = [dict(id=0, name=0)]

    videos_list = mmcv.list_from_file(osp.join(ann_dir, split, 'list.txt'))
    for video_name in tqdm(videos_list, desc=split):
        video = dict(id=records['vid_id'], name=video_name)
        got10k['videos'].append(video)

        video_path = osp.join(ann_dir, split, video_name)
        ann_file = osp.join(video_path, 'groundtruth.txt')
        gt_bboxes = mmcv.list_from_file(ann_file)

        img_files = glob.glob(osp.join(video_path, '*.jpg'))
        img_files = sorted(
            img_files, key=lambda x: int(x.split(os.sep)[-1][:-4]))
        img = mmcv.imread(osp.join(video_path, '00000001.jpg'))
        height, width, _ = img.shape
        if split in ['train', 'val']:
            absence_label = mmcv.list_from_file(
                osp.join(video_path, 'absence.label'))
            # cover_label denotes the ranges of object visible ratios, ant it's
            # in range [0,8] which correspond to ranges of object visible
            # ratios: 0%, (0%, 15%], (15%~30%], (30%, 45%], (45%, 60%],
            # (60%, 75%], (75%, 90%], (90%, 100%) and 100% respectively
            cover_label = mmcv.list_from_file(
                osp.join(video_path, 'cover.label'))
            # cut_by_image_label denotes whether the object is cut by the image
            # boundary.
            cut_by_image_label = mmcv.list_from_file(
                osp.join(video_path, 'cut_by_image.label'))
        for frame_id, img_file in enumerate(img_files):
            img_name = img_file.split(os.sep)[-1]
            # the images' root is not included in file_name
            file_name = osp.join(video_name, img_name)
            image = dict(
                file_name=file_name,
                height=height,
                width=width,
                id=records['img_id'],
                frame_id=frame_id,
                video_id=records['vid_id'])
            got10k['images'].append(image)

            ann = dict(
                id=records['ann_id'],
                video_id=records['vid_id'],
                image_id=records['img_id'],
                instance_id=records['global_instance_id'],
                category_id=0)

            if split == 'test':
                if frame_id == 0:
                    bbox = list(map(float, gt_bboxes[0].split(',')))
                else:
                    bbox = [0., 0., 0., 0.]
                ann.update(dict(bbox=bbox, area=bbox[2] * bbox[3]))
            else:
                bbox = list(map(float, gt_bboxes[frame_id].split(',')))
                ann.update(
                    dict(
                        bbox=bbox,
                        area=bbox[2] * bbox[3],
                        absence=absence_label[frame_id] == '1',
                        cover=int(cover_label[frame_id]),
                        cut_by_image=cut_by_image_label[frame_id] == '1'))

            got10k['annotations'].append(ann)

            records['ann_id'] += 1
            records['img_id'] += 1
        records['global_instance_id'] += 1
        records['vid_id'] += 1

    if not osp.isdir(save_dir):
        os.makedirs(save_dir)
    mmcv.dump(got10k, osp.join(save_dir, f'got10k_{split}.json'))
    print(f'-----GOT10k {split} Dataset------')
    print(f'{records["vid_id"]- 1} videos')
    print(f'{records["global_instance_id"]- 1} instances')
    print(f'{records["img_id"]- 1} images')
    print(f'{records["ann_id"] - 1} objects')
    print('-----------------------------')


def main():
    args = parse_args()
    if args.split == 'all':
        for split in ['train', 'val', 'test']:
            convert_got10k(args.input, args.output, split=split)
    else:
        convert_got10k(args.input, args.output, split=args.split)


if __name__ == '__main__':
    main()
