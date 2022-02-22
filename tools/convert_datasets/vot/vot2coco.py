# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
from collections import defaultdict

import cv2
import mmcv
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='VOT dataset to COCO Video format')
    parser.add_argument(
        '-i',
        '--input',
        help='root directory of VOT dataset',
    )
    parser.add_argument(
        '-o',
        '--output',
        help='directory to save coco formatted label file',
    ),
    parser.add_argument(
        '--dataset_type',
        help='the type of vot challenge',
        default='vot2018',
        choices=[
            'vot2018', 'vot2018_lt', 'vot2019', 'vot2019_lt', 'vot2019_rgbd',
            'vot2019_rgbt'
        ])
    return parser.parse_args()


def parse_attribute(video_path, attr_name, img_num):
    """Parse attribute of each video in VOT.

    Args:
        video_path (str): The path of video.
        attr_name (str): The name of video's attribute.
        img_num (str): The length of video.

    Returns:
        attr_list (list): The element is the tag of each image.
    """
    attr_path = osp.join(video_path, attr_name + '.tag')
    if osp.isfile(attr_path):
        attr_list = mmcv.list_from_file(attr_path)
    else:
        attr_list = []
    # unspecified tag is '0'(default)
    attr_list += ['0'] * (img_num - len(attr_list))
    return attr_list


def convert_vot(ann_dir, save_dir, dataset_type):
    """Convert vot dataset to COCO style.

    Args:
        ann_dir (str): The path of vot dataset
        save_dir (str): The path to save `vot`.
        dataset_type (str): The type of vot challenge.
    """
    vot = defaultdict(list)
    records = dict(vid_id=1, img_id=1, ann_id=1, global_instance_id=1)
    vot['categories'] = [dict(id=0, name=0)]

    videos_list = os.listdir(osp.join(ann_dir, 'data'))
    videos_list = [
        x for x in videos_list if osp.isdir(osp.join(ann_dir, 'data', x))
    ]
    for video_name in tqdm(videos_list):
        video = dict(id=records['vid_id'], name=video_name)
        vot['videos'].append(video)

        video_path = osp.join(ann_dir, 'data', video_name)
        ann_file = osp.join(video_path, 'groundtruth.txt')
        gt_anns = mmcv.list_from_file(ann_file)

        camera_motion = parse_attribute(video_path, 'camera_motion',
                                        len(gt_anns))
        illustration_change = parse_attribute(video_path, 'illu_change',
                                              len(gt_anns))
        motion_change = parse_attribute(video_path, 'motion_change',
                                        len(gt_anns))
        occlusion = parse_attribute(video_path, 'occlusion', len(gt_anns))
        size_change = parse_attribute(video_path, 'size_change', len(gt_anns))

        img = mmcv.imread(osp.join(video_path, 'color', '00000001.jpg'))
        height, width, _ = img.shape
        for frame_id, gt_anno in enumerate(gt_anns):
            file_name = '%08d' % (frame_id + 1) + '.jpg'
            file_name = osp.join(video_name, 'color', file_name)
            image = dict(
                file_name=file_name,
                height=height,
                width=width,
                id=records['img_id'],
                frame_id=frame_id,
                video_id=records['vid_id'])
            vot['images'].append(image)

            ann = dict(
                id=records['ann_id'],
                video_id=records['vid_id'],
                image_id=records['img_id'],
                instance_id=records['global_instance_id'],
                category_id=0,
                camera_motion=camera_motion[frame_id] == '1',
                illustration_change=illustration_change[frame_id] == '1',
                motion_change=motion_change[frame_id] == '1',
                occlusion=occlusion[frame_id] == '1',
                size_change=size_change[frame_id] == '1')

            anno = gt_anno.split(',')
            # TODO support mask annotations after VOT2019
            if anno[0][0] == 'm':
                continue
            else:
                # bbox is in [x1, y1, x2, y2, x3, y3, x4, y4] format
                bbox = list(map(lambda x: float(x), anno))
                if len(bbox) == 4:
                    bbox = [
                        bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1],
                        bbox[0] + bbox[2], bbox[1] + bbox[3], bbox[0],
                        bbox[1] + bbox[3]
                    ]
                assert len(bbox) == 8
                ann['bbox'] = bbox
                ann['area'] = cv2.contourArea(
                    np.array(bbox, dtype='int').reshape(4, 2))

            vot['annotations'].append(ann)

            records['ann_id'] += 1
            records['img_id'] += 1
        records['global_instance_id'] += 1
        records['vid_id'] += 1

    if not osp.isdir(save_dir):
        os.makedirs(save_dir)
    mmcv.dump(vot, osp.join(save_dir, f'{dataset_type}.json'))
    print(f'-----VOT Challenge {dataset_type} Dataset------')
    print(f'{records["vid_id"]- 1} videos')
    print(f'{records["global_instance_id"]- 1} instances')
    print(f'{records["img_id"]- 1} images')
    print(f'{records["ann_id"] - 1} objects')
    print('-----------------------------')


def main():
    args = parse_args()
    convert_vot(args.input, args.output, args.dataset_type)


if __name__ == '__main__':
    main()
