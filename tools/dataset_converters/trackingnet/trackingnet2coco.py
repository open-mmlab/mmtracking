# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
from collections import defaultdict

import mmcv
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='TrackingNet test dataset to COCO Video format')
    parser.add_argument(
        '-i',
        '--input',
        help='root directory of TrackingNet test dataset',
    )
    parser.add_argument(
        '-o',
        '--output',
        help='directory to save coco formatted label file',
    )
    parser.add_argument(
        '--split',
        help="the split set of trackingnet,'all' denotes the whole dataset",
        choices=['train', 'test', 'all'],
        default='all')
    return parser.parse_args()


def convert_trackingnet(ann_dir, save_dir, split='test'):
    """Convert trackingnet dataset to COCO style.

    Args:
        ann_dir (str): The path of trackingnet test dataset
        save_dir (str): The path to save `trackingnet`.
        split (str): the split ('train' or 'test') of dataset.
    """
    if split == 'test':
        chunks = ['TEST']
    elif split == 'train':
        chunks = [f'TRAIN_{i}' for i in range(12)]
    else:
        NotImplementedError

    trackingnet = defaultdict(list)
    records = dict(vid_id=1, img_id=1, ann_id=1, global_instance_id=1)
    trackingnet['categories'] = [dict(id=0, name=0)]

    for chunk in chunks:
        chunk_ann_dir = osp.join(ann_dir, chunk)
        assert osp.isdir(
            chunk_ann_dir
        ), f'annotation directory {chunk_ann_dir} does not exist'

        videos_list = os.listdir(osp.join(chunk_ann_dir, 'frames'))
        for video_name in tqdm(videos_list, desc=f'[{chunk}]'):
            video = dict(id=records['vid_id'], name=video_name)
            trackingnet['videos'].append(video)

            ann_file = osp.join(chunk_ann_dir, 'anno', video_name + '.txt')
            gt_bboxes = mmcv.list_from_file(ann_file)
            video_path = osp.join(chunk_ann_dir, 'frames', video_name)
            img_names = os.listdir(video_path)
            img_names = sorted(img_names, key=lambda x: int(x[:-4]))
            img = mmcv.imread(osp.join(video_path, '0.jpg'))
            height, width, _ = img.shape
            for frame_id, img_name in enumerate(img_names):
                file_name = '%d' % (frame_id) + '.jpg'
                assert img_name == file_name
                # the images' root is not included in file_name
                file_name = osp.join(chunk, 'frames', video_name, img_name)
                image = dict(
                    file_name=file_name,
                    height=height,
                    width=width,
                    id=records['img_id'],
                    frame_id=frame_id,
                    video_id=records['vid_id'])
                trackingnet['images'].append(image)

                if split == 'test':
                    if frame_id == 0:
                        bbox = list(map(float, gt_bboxes[0].split(',')))
                    else:
                        bbox = [0., 0., 0., 0.]
                else:
                    bbox = list(map(float, gt_bboxes[frame_id].split(',')))
                ann = dict(
                    id=records['ann_id'],
                    video_id=records['vid_id'],
                    image_id=records['img_id'],
                    instance_id=records['global_instance_id'],
                    category_id=0,
                    bbox=bbox,
                    area=bbox[2] * bbox[3])
                trackingnet['annotations'].append(ann)

                records['ann_id'] += 1
                records['img_id'] += 1
            records['global_instance_id'] += 1
            records['vid_id'] += 1

    if not osp.isdir(save_dir):
        os.makedirs(save_dir)
    mmcv.dump(trackingnet, osp.join(save_dir, f'trackingnet_{split}.json'))
    print(f'-----TrackingNet {split} Dataset------')
    print(f'{records["vid_id"]- 1} videos')
    print(f'{records["global_instance_id"]- 1} instances')
    print(f'{records["img_id"]- 1} images')
    print(f'{records["ann_id"] - 1} objects')
    print('-----------------------------')


def main():
    args = parse_args()
    if args.split == 'all':
        for split in ['train', 'test']:
            convert_trackingnet(args.input, args.output, split=split)
    else:
        convert_trackingnet(args.input, args.output, split=args.split)


if __name__ == '__main__':
    main()
