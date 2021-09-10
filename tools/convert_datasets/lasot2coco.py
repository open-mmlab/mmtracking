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
    return parser.parse_args()


def convert_lasot_test(lasot_test, ann_dir, save_dir):
    """Convert lasot dataset to COCO style.

    Args:
        lasot_test (dict): The converted COCO style annotations.
        ann_dir (str): The path of lasot test dataset
        save_dir (str): The path to save `lasot_test`.
    """
    records = dict(vid_id=1, img_id=1, ann_id=1, global_instance_id=1)
    videos_list = osp.join(ann_dir, 'testing_set.txt')
    videos_list = mmcv.list_from_file(videos_list)

    lasot_test['categories'] = [dict(id=0, name=0)]

    for video_name in tqdm(videos_list):
        video_path = osp.join(ann_dir, video_name)
        video = dict(id=records['vid_id'], name=video_name)
        lasot_test['videos'].append(video)

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
            file_name = osp.join(video_name, 'img', file_name)
            image = dict(
                file_name=file_name,
                height=height,
                width=width,
                id=records['img_id'],
                frame_id=frame_id,
                video_id=records['vid_id'])
            lasot_test['images'].append(image)

            x1, y1, w, h = gt_bbox.split(',')
            ann = dict(
                id=records['ann_id'],
                image_id=records['img_id'],
                instance_id=records['global_instance_id'],
                category_id=0,
                bbox=[int(x1), int(y1), int(w),
                      int(h)],
                area=int(w) * int(h),
                full_occlusion=full_occlusion[frame_id] == '1',
                out_of_view=out_of_view[frame_id] == '1')
            lasot_test['annotations'].append(ann)

            records['ann_id'] += 1
            records['img_id'] += 1
        records['global_instance_id'] += 1
        records['vid_id'] += 1

    if not osp.isdir(save_dir):
        os.makedirs(save_dir)
    mmcv.dump(lasot_test, osp.join(save_dir, 'lasot_test.json'))
    print('-----LaSOT Test Dataset------')
    print(f'{records["vid_id"]- 1} videos')
    print(f'{records["global_instance_id"]- 1} instances')
    print(f'{records["img_id"]- 1} images')
    print(f'{records["ann_id"] - 1} objects')
    print('-----------------------------')


def main():
    args = parse_args()
    lasot_test = defaultdict(list)
    convert_lasot_test(lasot_test, args.input, args.output)


if __name__ == '__main__':
    main()
