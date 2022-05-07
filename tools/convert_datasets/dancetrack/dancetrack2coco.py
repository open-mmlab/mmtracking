# Copyright (c) OpenMMLab. All rights reserved.
# This script converts DanceTrack labels into COCO style.
# Official repo of the DanceTrack dataset:
# https://github.com/DanceTrack/DanceTrack
#
# Label format of DanceTrack dataset:
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
# Classes in DanceTrack:
#   1: 'pedestrian'
#
#   USELESS classes are not included into the json file.
#   IGNORES classes are included with `ignore=True`.
#
# This file is adapted from the data label conversion file for MOT
# But as Dancetrack does not provide public detections and provides
# official train/val/test splitting, we make necessary adaptation.

import argparse
import os
import os.path as osp
from collections import defaultdict

import mmcv
from tqdm import tqdm

USELESS = [3, 4, 5, 6, 9, 10, 11]
IGNORES = [2, 7, 8, 12, 13]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert DanceTrack label and detections to \
        COCO-VID format.')
    parser.add_argument('-i', '--input', help='path of MOT data')
    parser.add_argument(
        '-o', '--output', help='path to save coco formatted label file')
    return parser.parse_args()


def parse_gts(gts):
    outputs = defaultdict(list)
    for gt in gts:
        gt = gt.strip().split(',')
        frame_id, ins_id = map(int, gt[:2])
        bbox = list(map(float, gt[2:6]))
        conf = float(gt[6])
        class_id = int(gt[7])
        visibility = float(gt[8])
        if class_id in USELESS:
            continue
        elif class_id in IGNORES:
            continue
        anns = dict(
            category_id=1,
            bbox=bbox,
            area=bbox[2] * bbox[3],
            iscrowd=False,
            visibility=visibility,
            mot_instance_id=ins_id,
            mot_conf=conf,
            mot_class_id=class_id)
        outputs[frame_id].append(anns)
    return outputs


def main():
    args = parse_args()
    if not osp.isdir(args.output):
        os.makedirs(args.output)

    sets = ['train', 'val', 'test']
    vid_id, img_id, ann_id = 1, 1, 1

    for subset in sets:
        ins_id = 0
        print(f'Converting {subset} set to COCO format')
        in_folder = osp.join(args.input, subset)
        out_file = osp.join(args.output, f'{subset}_cocoformat.json')
        outputs = defaultdict(list)
        outputs['categories'] = [dict(id=1, name='pedestrian')]

        video_names = os.listdir(in_folder)
        video_names = [d for d in video_names if d != '.DS_Store']
        for video_name in tqdm(video_names):
            # basic params
            parse_gt = 'test' not in subset
            ins_maps = dict()
            # load video infos
            video_folder = osp.join(in_folder, video_name)
            infos = mmcv.list_from_file(f'{video_folder}/seqinfo.ini')
            # video-level infos
            assert video_name == infos[1].strip().split('=')[1]
            img_folder = infos[2].strip().split('=')[1]
            img_names = os.listdir(f'{video_folder}/{img_folder}')
            img_names = [d for d in img_names if d != '.DS_Store']
            img_names = sorted(img_names)
            fps = int(infos[3].strip().split('=')[1])
            num_imgs = int(infos[4].strip().split('=')[1])

            assert num_imgs == len(img_names)
            width = int(infos[5].strip().split('=')[1])
            height = int(infos[6].strip().split('=')[1])
            video = dict(
                id=vid_id,
                name=video_name,
                fps=fps,
                width=width,
                height=height)
            # parse annotations
            if parse_gt:
                gts = mmcv.list_from_file(f'{video_folder}/gt/gt.txt')
                img2gts = parse_gts(gts)

            # image and box level infos
            for frame_id, name in enumerate(img_names):
                img_name = osp.join(video_name, img_folder, name)
                mot_frame_id = int(name.split('.')[0])
                image = dict(
                    id=img_id,
                    video_id=vid_id,
                    file_name=img_name,
                    height=height,
                    width=width,
                    frame_id=frame_id,
                    mot_frame_id=mot_frame_id)
                if parse_gt:
                    gts = img2gts[mot_frame_id]
                    for gt in gts:
                        gt.update(id=ann_id, image_id=img_id)
                        mot_ins_id = gt['mot_instance_id']
                        if mot_ins_id in ins_maps:
                            gt['instance_id'] = ins_maps[mot_ins_id]
                        else:
                            gt['instance_id'] = ins_id
                            ins_maps[mot_ins_id] = ins_id
                            ins_id += 1
                        outputs['annotations'].append(gt)
                        ann_id += 1

                outputs['images'].append(image)
                img_id += 1
            outputs['videos'].append(video)
            vid_id += 1
            outputs['num_instances'] = ins_id
        print(f'{subset} has {ins_id} instances.')
        mmcv.dump(outputs, out_file)
        print(f'Done! Saved as {out_file}')


if __name__ == '__main__':
    main()
