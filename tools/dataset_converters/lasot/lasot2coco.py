# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
from collections import defaultdict

import mmcv
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='LaSOT test dataset to COCO Video format')
    parser.add_argument(
        '-i',
        '--input',
        help='root directory of LaSOT test dataset',
    )
    parser.add_argument(
        '-o',
        '--output',
        help='directory to save coco formatted label file',
    )
    parser.add_argument(
        '--split',
        help='the split set of lasot, all denotes the whole dataset',
        choices=['train', 'test', 'all'],
        default='all')
    return parser.parse_args()


def convert_lasot(ann_dir, save_dir, split='test'):
    """Convert lasot dataset to COCO style.

    Args:
        ann_dir (str): The path of lasot dataset
        save_dir (str): The path to save `lasot`.
        split (str): the split ('train' or 'test') of dataset.
    """
    assert split in ['train', 'test'], f'split [{split}] does not exist'
    lasot = defaultdict(list)
    records = dict(vid_id=1, img_id=1, ann_id=1, global_instance_id=1)
    lasot['categories'] = [dict(id=0, name=0)]
    videos_list = mmcv.list_from_file(
        osp.join(osp.dirname(__file__), 'testing_set.txt'))
    if split == 'train':
        train_videos_list = []
        for video_class in os.listdir(ann_dir):
            for video_id in os.listdir(osp.join(ann_dir, video_class)):
                if video_id not in videos_list:
                    train_videos_list.append(video_id)
        videos_list = train_videos_list

    for video_name in tqdm(videos_list, desc=split):
        video_class = video_name.split('-')[0]
        video_path = osp.join(ann_dir, video_class, video_name)
        video = dict(id=records['vid_id'], name=video_name)
        lasot['videos'].append(video)

        gt_bboxes = mmcv.list_from_file(
            osp.join(video_path, 'groundtruth.txt'))
        full_occlusion = mmcv.list_from_file(
            osp.join(video_path, 'full_occlusion.txt'))
        full_occlusion = full_occlusion[0].split(',')
        out_of_view = mmcv.list_from_file(
            osp.join(video_path, 'out_of_view.txt'))
        out_of_view = out_of_view[0].split(',')

        img = mmcv.imread(osp.join(video_path, 'img/00000001.jpg'))
        height, width, _ = img.shape
        for frame_id, gt_bbox in enumerate(gt_bboxes):
            file_name = '%08d' % (frame_id + 1) + '.jpg'
            file_name = osp.join(video_class, video_name, 'img', file_name)
            image = dict(
                file_name=file_name,
                height=height,
                width=width,
                id=records['img_id'],
                frame_id=frame_id,
                video_id=records['vid_id'])
            lasot['images'].append(image)

            x1, y1, w, h = gt_bbox.split(',')
            ann = dict(
                id=records['ann_id'],
                video_id=records['vid_id'],
                image_id=records['img_id'],
                instance_id=records['global_instance_id'],
                category_id=0,
                bbox=[int(x1), int(y1), int(w),
                      int(h)],
                area=int(w) * int(h),
                full_occlusion=full_occlusion[frame_id] == '1',
                out_of_view=out_of_view[frame_id] == '1')
            lasot['annotations'].append(ann)

            records['ann_id'] += 1
            records['img_id'] += 1
        records['global_instance_id'] += 1
        records['vid_id'] += 1

    if not osp.isdir(save_dir):
        os.makedirs(save_dir)
    mmcv.dump(lasot, osp.join(save_dir, f'lasot_{split}.json'))
    print(f'-----LaSOT {split} Dataset------')
    print(f'{records["vid_id"]- 1} videos')
    print(f'{records["global_instance_id"]- 1} instances')
    print(f'{records["img_id"]- 1} images')
    print(f'{records["ann_id"] - 1} objects')
    print('-----------------------------')


def main():
    args = parse_args()
    if args.split == 'all':
        for split in ['train', 'test']:
            convert_lasot(args.input, args.output, split=split)
    else:
        convert_lasot(args.input, args.output, split=args.split)


if __name__ == '__main__':
    main()
