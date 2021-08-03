import argparse
import glob
import os
import os.path as osp
from collections import defaultdict

import mmcv
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='OTB2015 dataset to COCO Video format')
    parser.add_argument(
        '-i',
        '--input',
        help='root directory of OTB2015 dataset',
    )
    parser.add_argument(
        '-o',
        '--output',
        help='directory to save coco formatted label file',
    )
    return parser.parse_args()


def convert_otb2015(otb, ann_dir, save_dir):
    """Convert OTB2015 dataset to COCO style.

    Args:
        lasot_test (dict): The converted COCO style annotations.
        ann_dir (str): The path of OTB2015 dataset
        save_dir (str): The path to save `OTB2015`.
    """
    records = dict(vid_id=1, img_id=1, ann_id=1, global_instance_id=1)
    videos_list = os.listdir(ann_dir)
    otb['categories'] = [dict(id=0, name=0)]

    for video_name in tqdm(videos_list):
        video_path = osp.join(ann_dir, video_name)
        video = dict(id=records['vid_id'], name=video_name)
        otb['videos'].append(video)

        if video_name == 'David':
            start_frame_id = 300
        else:
            start_frame_id = 1

        # img_list = glob.glob(osp.join(video_path, 'img', '*.jpg'))
        img_list = os.listdir(osp.join(video_path, 'img'))
        img_list = sorted(img_list)

        img = mmcv.imread(osp.join(video_path, 'img', img_list[0]))
        height, width, _ = img.shape

        gt_list = glob.glob(
            osp.join(ann_dir, video_name, 'groundtruth_rect*.txt'))
        for gt_file in gt_list:
            gt_bboxes = mmcv.list_from_file(gt_file)
            for i, gt_bbox in enumerate(gt_bboxes):
                frame_id = i + start_frame_id - 1
                file_name = osp.join(video_name, 'img', img_list[frame_id])
                image = dict(
                    file_name=file_name,
                    height=height,
                    width=width,
                    id=records['img_id'],
                    frame_id=i,
                    video_id=records['vid_id'])
                otb['images'].append(image)

                x1, y1, w, h = gt_bbox.split(',')
                anno_dict = dict(
                    id=records['ann_id'],
                    image_id=records['img_id'],
                    instance_id=records['global_instance_id'],
                    category_id=0,
                    bbox=[int(x1), int(y1), int(w),
                          int(h)],
                    area=int(w) * int(h),
                )
                otb['annotations'].append(anno_dict)

                records['ann_id'] += 1
                records['img_id'] += 1

            records['global_instance_id'] += 1
            records['vid_id'] += 1

    mmcv.dump(otb, osp.join(save_dir, 'otb2015.json'))
    print('-----OTB2015 Dataset------')
    print(f'{records["vid_id"]- 1} videos')
    print(f'{records["global_instance_id"]- 1} instances')
    print(f'{records["img_id"]- 1} images')
    print(f'{records["ann_id"] - 1} objects')
    print('-----------------------------')


def main():
    args = parse_args()
    otb = defaultdict(list)
    convert_otb2015(otb, args.input, args.output)


if __name__ == '__main__':
    main()
