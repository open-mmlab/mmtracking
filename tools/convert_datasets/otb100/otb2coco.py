# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import os
import os.path as osp
import re
from collections import defaultdict

import mmcv
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='OTB100 dataset to COCO Video format')
    parser.add_argument(
        '-i',
        '--input',
        help='root directory of OTB100 dataset',
    )
    parser.add_argument(
        '-o',
        '--output',
        help='directory to save coco formatted label file',
    )
    return parser.parse_args()


def convert_otb100(otb, ann_dir, save_dir):
    """Convert OTB100 dataset to COCO style.

    Args:
        otb (dict): The converted COCO style annotations.
        ann_dir (str): The path of OTB100 dataset
        save_dir (str): The path to save `OTB100`.
    """
    records = dict(vid_id=1, img_id=1, ann_id=1, global_instance_id=1)
    ann_dir = osp.join(ann_dir, 'data')
    videos_list = os.listdir(ann_dir)
    otb['categories'] = [dict(id=0, name=0)]

    for video_name in tqdm(videos_list):
        video_path = osp.join(ann_dir, video_name)

        if video_name == 'David':
            start_frame_id = 300
        # The first five frames in Tiger1 can not be used
        # as the initinal frames.
        # Details can be seen in tracker_benchmark_v1.0/initOmit/tiger1.txt.
        # The start_frame_id is 1-based.
        elif video_name == 'Tiger1':
            start_frame_id = 6
        else:
            start_frame_id = 1

        img_list = os.listdir(osp.join(video_path, 'img'))
        img_list = sorted(img_list)

        img = mmcv.imread(osp.join(video_path, 'img', img_list[0]))
        height, width, _ = img.shape

        # One video may have several tracking instances with their
        # respective annotations.
        gt_list = glob.glob(
            osp.join(ann_dir, video_name, 'groundtruth_rect*.txt'))
        for gt_file in gt_list:
            # exclude empty files
            if osp.getsize(gt_file) == 0:
                continue

            video = dict(id=records['vid_id'], name=video_name)
            otb['videos'].append(video)

            gt_bboxes = mmcv.list_from_file(gt_file)
            if video_name == 'Tiger1':
                gt_bboxes = gt_bboxes[start_frame_id - 1:]
            for frame_id, gt_bbox in enumerate(gt_bboxes):
                src_frame_id = frame_id + start_frame_id - 1
                file_name = osp.join(video_name, 'img', img_list[src_frame_id])
                image = dict(
                    file_name=file_name,
                    height=height,
                    width=width,
                    id=records['img_id'],
                    frame_id=frame_id,
                    video_id=records['vid_id'])
                otb['images'].append(image)

                bbox = list(map(int, re.findall(r'-?\d+', gt_bbox)))
                assert len(bbox) == 4
                anno_dict = dict(
                    id=records['ann_id'],
                    video_id=records['vid_id'],
                    image_id=records['img_id'],
                    instance_id=records['global_instance_id'],
                    category_id=0,
                    bbox=bbox,
                    area=bbox[2] * bbox[3],
                )
                otb['annotations'].append(anno_dict)

                records['ann_id'] += 1
                records['img_id'] += 1

            records['global_instance_id'] += 1
            records['vid_id'] += 1

    if not osp.isdir(save_dir):
        os.makedirs(save_dir)
    mmcv.dump(otb, osp.join(save_dir, 'otb100.json'))
    print('-----OTB100 Dataset------')
    print(f'{records["vid_id"]- 1} videos')
    print(f'{records["global_instance_id"]- 1} instances')
    print(f'{records["img_id"]- 1} images')
    print(f'{records["ann_id"] - 1} objects')
    print('-----------------------------')


def main():
    args = parse_args()
    otb = defaultdict(list)
    convert_otb100(otb, args.input, args.output)


if __name__ == '__main__':
    main()
