# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import xml.etree.ElementTree as ET
from collections import defaultdict

import mmcv
from tqdm import tqdm

CLASSES = ('airplane', 'antelope', 'bear', 'bicycle', 'bird', 'bus', 'car',
           'cattle', 'dog', 'domestic_cat', 'elephant', 'fox', 'giant_panda',
           'hamster', 'horse', 'lion', 'lizard', 'monkey', 'motorcycle',
           'rabbit', 'red_panda', 'sheep', 'snake', 'squirrel', 'tiger',
           'train', 'turtle', 'watercraft', 'whale', 'zebra')

CLASSES_ENCODES = ('n02691156', 'n02419796', 'n02131653', 'n02834778',
                   'n01503061', 'n02924116', 'n02958343', 'n02402425',
                   'n02084071', 'n02121808', 'n02503517', 'n02118333',
                   'n02510455', 'n02342885', 'n02374451', 'n02129165',
                   'n01674464', 'n02484322', 'n03790512', 'n02324045',
                   'n02509815', 'n02411705', 'n01726692', 'n02355227',
                   'n02129604', 'n04468005', 'n01662784', 'n04530566',
                   'n02062744', 'n02391049')

cats_id_maps = {}
for k, v in enumerate(CLASSES_ENCODES, 1):
    cats_id_maps[v] = k


def parse_args():
    parser = argparse.ArgumentParser(
        description='ImageNet VID to COCO Video format')
    parser.add_argument(
        '-i',
        '--input',
        help='root directory of ImageNet VID annotations',
    )
    parser.add_argument(
        '-o',
        '--output',
        help='directory to save coco formatted label file',
    )
    return parser.parse_args()


def parse_train_list(ann_dir):
    """Parse the txt file of ImageNet VID train dataset."""
    img_list = osp.join(ann_dir, 'Lists/VID_train_15frames.txt')
    img_list = mmcv.list_from_file(img_list)
    train_infos = defaultdict(list)
    for info in img_list:
        info = info.split(' ')
        if info[0] not in train_infos:
            train_infos[info[0]] = dict(
                vid_train_frames=[int(info[2]) - 1], num_frames=int(info[-1]))
        else:
            train_infos[info[0]]['vid_train_frames'].append(int(info[2]) - 1)
    return train_infos


def parse_val_list(ann_dir):
    """Parse the txt file of ImageNet VID val dataset."""
    img_list = osp.join(ann_dir, 'Lists/VID_val_videos.txt')
    img_list = mmcv.list_from_file(img_list)
    val_infos = defaultdict(list)
    for info in img_list:
        info = info.split(' ')
        val_infos[info[0]] = dict(num_frames=int(info[-1]))
    return val_infos


def convert_vid(VID, ann_dir, save_dir, mode='train'):
    """Convert ImageNet VID dataset in COCO style.

    Args:
        VID (dict): The converted COCO style annotations.
        ann_dir (str): The path of ImageNet VID dataset.
        save_dir (str): The path to save `VID`.
        mode (str): Convert train dataset or validation dataset. Options are
            'train', 'val'. Default: 'train'.
    """
    assert mode in ['train', 'val']
    records = dict(
        vid_id=1,
        img_id=1,
        ann_id=1,
        global_instance_id=1,
        num_vid_train_frames=0,
        num_no_objects=0)
    obj_num_classes = dict()
    xml_dir = osp.join(ann_dir, 'Annotations/VID/')
    if mode == 'train':
        vid_infos = parse_train_list(ann_dir)
    else:
        vid_infos = parse_val_list(ann_dir)
    for vid_info in tqdm(vid_infos):
        instance_id_maps = dict()
        vid_train_frames = vid_infos[vid_info].get('vid_train_frames', [])
        records['num_vid_train_frames'] += len(vid_train_frames)
        video = dict(
            id=records['vid_id'],
            name=vid_info,
            vid_train_frames=vid_train_frames)
        VID['videos'].append(video)
        num_frames = vid_infos[vid_info]['num_frames']
        for frame_id in range(num_frames):
            is_vid_train_frame = True if frame_id in vid_train_frames \
                else False
            img_prefix = osp.join(vid_info, '%06d' % frame_id)
            xml_name = osp.join(xml_dir, f'{img_prefix}.xml')
            # parse XML annotation file
            tree = ET.parse(xml_name)
            root = tree.getroot()
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            image = dict(
                file_name=f'{img_prefix}.JPEG',
                height=height,
                width=width,
                id=records['img_id'],
                frame_id=frame_id,
                video_id=records['vid_id'],
                is_vid_train_frame=is_vid_train_frame)
            VID['images'].append(image)
            if root.findall('object') == []:
                print(xml_name, 'has no objects.')
                records['num_no_objects'] += 1
                records['img_id'] += 1
                continue
            for obj in root.findall('object'):
                name = obj.find('name').text
                if name not in cats_id_maps:
                    continue
                category_id = cats_id_maps[name]
                bnd_box = obj.find('bndbox')
                x1, y1, x2, y2 = [
                    int(bnd_box.find('xmin').text),
                    int(bnd_box.find('ymin').text),
                    int(bnd_box.find('xmax').text),
                    int(bnd_box.find('ymax').text)
                ]
                w = x2 - x1
                h = y2 - y1
                track_id = obj.find('trackid').text
                if track_id in instance_id_maps:
                    instance_id = instance_id_maps[track_id]
                else:
                    instance_id = records['global_instance_id']
                    records['global_instance_id'] += 1
                    instance_id_maps[track_id] = instance_id
                occluded = obj.find('occluded').text
                generated = obj.find('generated').text
                ann = dict(
                    id=records['ann_id'],
                    video_id=records['vid_id'],
                    image_id=records['img_id'],
                    category_id=category_id,
                    instance_id=instance_id,
                    bbox=[x1, y1, w, h],
                    area=w * h,
                    iscrowd=False,
                    occluded=occluded == '1',
                    generated=generated == '1')
                if category_id not in obj_num_classes:
                    obj_num_classes[category_id] = 1
                else:
                    obj_num_classes[category_id] += 1
                VID['annotations'].append(ann)
                records['ann_id'] += 1
            records['img_id'] += 1
        records['vid_id'] += 1
    if not osp.isdir(save_dir):
        os.makedirs(save_dir)
    mmcv.dump(VID, osp.join(save_dir, f'imagenet_vid_{mode}.json'))
    print(f'-----ImageNet VID {mode}------')
    print(f'{records["vid_id"]- 1} videos')
    print(f'{records["img_id"]- 1} images')
    print(
        f'{records["num_vid_train_frames"]} train frames for video detection')
    print(f'{records["num_no_objects"]} images have no objects')
    print(f'{records["ann_id"] - 1} objects')
    print('-----------------------')
    for i in range(1, len(CLASSES) + 1):
        print(f'Class {i} {CLASSES[i - 1]} has {obj_num_classes[i]} objects.')


def main():
    args = parse_args()

    categories = []
    for k, v in enumerate(CLASSES, 1):
        categories.append(
            dict(id=k, name=v, encode_name=CLASSES_ENCODES[k - 1]))

    VID_train = defaultdict(list)
    VID_train['categories'] = categories
    convert_vid(VID_train, args.input, args.output, 'train')

    VID_val = defaultdict(list)
    VID_val['categories'] = categories
    convert_vid(VID_val, args.input, args.output, 'val')


if __name__ == '__main__':
    main()
