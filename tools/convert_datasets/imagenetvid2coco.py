import argparse
import os.path as osp
import random
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
        description='ImageNet DET and VID to COCO Video format')
    parser.add_argument(
        '-d',
        '--ann_dir',
        help='root directory of BDD label Json files',
    )
    parser.add_argument(
        '-s',
        '--save_dir',
        help='directory to save coco formatted label file',
    )
    return parser.parse_args()


def parse_train_list(ann_dir):
    file = 'Lists/VID_train_15frames.txt'
    img_list = osp.join(ann_dir, file)
    img_list = mmcv.list_from_file(img_list)
    train_infos = defaultdict(list)
    for info in img_list:
        info = info.split(' ')
        if info[0] not in train_infos.keys():
            train_infos[info[0]] = dict(
                key_frames=[int(info[2]) - 1], num_frames=int(info[-1]))
        else:
            train_infos[info[0]]['key_frames'].append(int(info[2]) - 1)
    return train_infos


def parse_val_list(ann_dir):
    file = 'Lists/VID_val_videos.txt'
    img_list = osp.join(ann_dir, file)
    img_list = mmcv.list_from_file(img_list)
    random.shuffle(img_list)
    val_infos = defaultdict(list)
    for info in img_list:
        info = info.split(' ')
        val_infos[info[0]] = dict(num_frames=int(info[-1]))
    return val_infos


def convert_vid(VID, ann_dir, save_dir, mode='train'):
    assert mode in ['train', 'val']
    records = dict(
        vid_id=0,
        img_id=0,
        ann_id=0,
        global_instance_id=0,
        num_key_frames=0,
        num_no_objects=0)
    obj_num_classes = dict()
    xml_dir = osp.join(ann_dir, 'Annotations/VID/')
    if mode == 'train':
        vid_infos = parse_train_list(ann_dir)
    else:
        vid_infos = parse_val_list(ann_dir)
    for vid_info in tqdm(vid_infos.keys()):
        instance_id_maps = dict()
        key_frames = vid_infos[vid_info].get('key_frames', [])
        records['num_key_frames'] += len(key_frames)
        video = dict(
            id=records['vid_id'], name=vid_info, key_frames=key_frames)
        VID['videos'].append(video)
        num_frames = vid_infos[vid_info]['num_frames']
        for frame_id in range(num_frames):
            is_key_frame = True if frame_id in key_frames else False
            img_prefix = vid_info + '/%06d' % frame_id
            xml_name = xml_dir + '/{}.xml'.format(img_prefix)
            tree = ET.parse(xml_name)
            root = tree.getroot()
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            image = dict(
                file_name='{}.JPEG'.format(img_prefix),
                height=height,
                width=width,
                id=records['img_id'],
                index=frame_id,
                video_id=records['vid_id'],
                key_frame=is_key_frame)
            VID['images'].append(image)
            if root.findall('object') == []:
                print(xml_name, 'has no objects.')
                records['num_no_objects'] += 1
                records['img_id'] += 1
                continue
            for obj in root.findall('object'):
                name = obj.find('name').text
                if name not in cats_id_maps.keys():
                    continue
                category_id = cats_id_maps[name]
                bnd_box = obj.find('bndbox')
                x1, y1, x2, y2 = [
                    int(bnd_box.find('xmin').text),
                    int(bnd_box.find('ymin').text),
                    int(bnd_box.find('xmax').text),
                    int(bnd_box.find('ymax').text)
                ]
                w = x2 - x1 + 1
                h = y2 - y1 + 1
                track_id = obj.find('trackid').text
                if track_id in instance_id_maps.keys():
                    instance_id = instance_id_maps[track_id]
                else:
                    instance_id = records['global_instance_id']
                    records['global_instance_id'] += 1
                    instance_id_maps[track_id] = instance_id
                occluded = obj.find('occluded').text
                generated = obj.find('generated').text
                ann = dict(
                    id=records['ann_id'],
                    image_id=records['img_id'],
                    category_id=category_id,
                    instance_id=instance_id,
                    bbox=[x1, y1, w, h],
                    area=w * h,
                    iscrowd=False,
                    ignore=False,
                    segmentation=[[x1, y1, x1, y2, x2, y2, x2, y1]],
                    is_occluded=occluded == '1',
                    generated=generated == '1')
                if category_id not in obj_num_classes.keys():
                    obj_num_classes[category_id] = 1
                else:
                    obj_num_classes[category_id] += 1
                VID['annotations'].append(ann)
                records['ann_id'] += 1
            records['img_id'] += 1
        records['vid_id'] += 1
    mmcv.dump(VID, osp.join(save_dir, 'ImageNet_VID_{}.json'.format(mode)))
    print('-----ImageNet VID {}------'.format(mode))
    print('{} videos'.format(records['vid_id']))
    print('{} images'.format(records['img_id']))
    print('{} key frames'.format(records['num_key_frames']))
    print('{} images have no objects'.format(records['num_no_objects']))
    print('{} objects'.format(records['ann_id']))
    print('-----------------------')
    # for k, v in obj_num_classes.items():
    #     print('Class {} {} has {} objects.'.format(k, CLASSES[k - 1], v))
    for i in range(1, 31):
        print('Class {} {} has {} objects.'.format(i, CLASSES[i - 1],
                                                   obj_num_classes[i]))


def main():
    args = parse_args()

    categories = []
    for k, v in enumerate(CLASSES, 1):
        categories.append(dict(id=k, name=v))

    VID = defaultdict(list)
    VID['categories'] = categories
    VID = convert_vid(VID, args.ann_dir, args.save_dir, 'train')

    VID = defaultdict(list)
    VID['categories'] = categories
    VID = convert_vid(VID, args.ann_dir, args.save_dir, 'val')


if __name__ == '__main__':
    main()
