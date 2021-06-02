# This script converts MOT labels into COCO style.
# Offical website of the MOT dataset: https://motchallenge.net/
#
# Label format of MOT dataset:
#   GTs:
#       <frame_id> # starts from 1 but COCO style starts from 0,
#       <instance_id>, <x1>, <y1>, <w>, <h>,
#       <conf> # conf is annotated as 0 if the object is ignored,
#       <class_id>, <visibility>
#
#   DETs and Results:
#       <frame_id>, <instance_id>, <x1>, <y1>, <w>, <h>, <conf>,
#       <x>, <y>, <z> # for 3D objects
#
# Classes in MOT:
#   1: 'pedestrian'
#   2: 'person on vehicle'
#   3: 'car'
#   4: 'bicycle'
#   5: 'motorbike'
#   6: 'non motorized vehicle'
#   7: 'static person'
#   8: 'distractor'
#   9: 'occluder'
#   10: 'occluder on the ground',
#   11: 'occluder full'
#   12: 'reflection'
#
#   USELESS classes are not included into the json file.
#   IGNORES classes are included with `ignore=True`.
import argparse
import os
import os.path as osp

import mmcv
import numpy as np
from tqdm import tqdm

USELESS = [3, 4, 5, 6, 9, 10, 11]
IGNORES = [2, 7, 8, 12, 13]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MOT label and detections to COCO-VID format.')
    parser.add_argument('-i', '--input', help='path of MOT data')
    parser.add_argument(
        '-o', '--output', help='path to save coco formatted label file')
    return parser.parse_args()


def main():
    args = parse_args()
    if not osp.exists(args.output):
        os.makedirs(args.output)
    elif os.listdir(args.output):
        raise OSError(f'Directory must empty: \'{args.output}\'')

    in_folder = osp.join(args.input, 'train')
    video_names = os.listdir(in_folder)
    sorted(video_names)
    for video_name in tqdm(video_names):
        # load video infos
        video_folder = osp.join(in_folder, video_name)
        infos = mmcv.list_from_file(f'{video_folder}/seqinfo.ini')
        # video-level infos
        assert video_name == infos[1].strip().split('=')[1]
        # img1
        raw_img_folder = infos[2].strip().split('=')[1]
        raw_img_names = os.listdir(f'{video_folder}/{raw_img_folder}')
        raw_img_names = sorted(raw_img_names)
        num_raw_imgs = int(infos[4].strip().split('=')[1])
        assert num_raw_imgs == len(raw_img_names)

        reid_train_folder = osp.join(args.output, 'train')
        if not osp.exists(reid_train_folder):
            os.makedirs(reid_train_folder)
        gts = mmcv.list_from_file(f'{video_folder}/gt/gt.txt')
        for gt in gts:
            gt = gt.strip().split(',')
            frame_id, ins_id = map(int, gt[:2])
            ltwh = list(map(float, gt[2:6]))
            class_id = int(gt[7])
            if class_id in USELESS:
                continue
            elif class_id in IGNORES:
                continue
            reid_img_folder = osp.join(reid_train_folder,
                                       f'{video_name}_{ins_id:06d}')
            if not osp.exists(reid_img_folder):
                os.makedirs(reid_img_folder)
            idx = len(os.listdir(reid_img_folder))
            reid_img_name = f'{idx:06d}.jpg'
            raw_img_name = raw_img_names[frame_id - 1]
            raw_img = mmcv.imread(
                f'{video_folder}/{raw_img_folder}/{raw_img_name}')
            xyxy = np.asarray(
                [ltwh[0], ltwh[1], ltwh[0] + ltwh[2], ltwh[1] + ltwh[3]])
            reid_img = mmcv.imcrop(raw_img, xyxy)
            mmcv.imwrite(reid_img, f'{reid_img_folder}/{reid_img_name}')

    # meta
    reid_meta_folder = osp.join(args.output, 'meta')
    if not osp.exists(reid_meta_folder):
        os.makedirs(reid_meta_folder)
    reid_val_list = []
    reid_img_folder_names = os.listdir(reid_train_folder)
    sorted(reid_img_folder_names)
    label = 0
    for reid_img_folder_name in reid_img_folder_names:
        reid_img_names = os.listdir(
            f'{reid_train_folder}/{reid_img_folder_name}')
        sorted(reid_img_names)
        for reid_img_name in reid_img_names:
            reid_val_list.append(
                f'{reid_img_folder_name}/{reid_img_name} {label}\n')
        label += 1
    with open(osp.join(reid_meta_folder, 'val.txt'), 'w') as f:
        f.writelines(reid_val_list)


if __name__ == '__main__':
    main()
