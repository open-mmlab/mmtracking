# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
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
           'train', 'turtle', 'watercraft', 'whale', 'zebra',
           'other_categeries')

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
        description='ImageNet DET to COCO Video format')
    parser.add_argument(
        '-i',
        '--input',
        help='root directory of ImageNet DET annotations',
    )
    parser.add_argument(
        '-o',
        '--output',
        help='directory to save coco formatted label file',
    )
    return parser.parse_args()


def parse_xml(img_name, xml_path, is_vid_train_frame, records, DET,
              obj_num_classes):
    """Parse xml annotations and record them.

    Args:
        img_name (str): image file path.
        xml_path (str): annotation file path.
        is_vid_train_frame (bool): If True, the image is used for the training
            of video object detection task, otherwise, not used.
        records (dict): The record information like image id, annotation id.
        DET (dict): The converted COCO style annotations.
        obj_num_classes (dict): The number of objects per class.

    Returns:
        tuple: (records, DET, obj_num_classes), records is the updated record
            information like image id, annotation id, DET is the updated
            COCO style annotations, obj_num_classes is the updated number of
            objects per class.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    image = dict(
        file_name=img_name,
        height=height,
        width=width,
        id=records['img_id'],
        is_vid_train_frame=is_vid_train_frame)
    DET['images'].append(image)
    if is_vid_train_frame:
        records['vid_train_frames'] += 1

    if root.findall('object') == []:
        print(f'{xml_path} has no objects.')
        records['num_no_objects'] += 1
        records['img_id'] += 1
        return records, DET, obj_num_classes

    for obj in root.findall('object'):
        name = obj.find('name').text
        if name in cats_id_maps:
            category_id = cats_id_maps[name]
        else:
            category_id = len(cats_id_maps) + 1
        bnd_box = obj.find('bndbox')
        x1, y1, x2, y2 = [
            int(bnd_box.find('xmin').text),
            int(bnd_box.find('ymin').text),
            int(bnd_box.find('xmax').text),
            int(bnd_box.find('ymax').text)
        ]
        w = x2 - x1
        h = y2 - y1
        ann = dict(
            id=records['ann_id'],
            image_id=records['img_id'],
            category_id=category_id,
            bbox=[x1, y1, w, h],
            area=w * h,
            iscrowd=False)
        DET['annotations'].append(ann)
        if category_id not in obj_num_classes:
            obj_num_classes[category_id] = 1
        else:
            obj_num_classes[category_id] += 1
        records['ann_id'] += 1
    records['img_id'] += 1
    return records, DET, obj_num_classes


def convert_det(DET, ann_dir, save_dir):
    """Convert ImageNet DET dataset in COCO style.

    Args:
        DET (dict): The converted COCO style annotations.
        ann_dir (str): The path of ImageNet DET dataset
        save_dir (str): The path to save `DET`.
    """
    dataset_sets = ('train/ILSVRC2013_train', 'train/ILSVRC2014_train_0000',
                    'train/ILSVRC2014_train_0001',
                    'train/ILSVRC2014_train_0002',
                    'train/ILSVRC2014_train_0003',
                    'train/ILSVRC2014_train_0004',
                    'train/ILSVRC2014_train_0005',
                    'train/ILSVRC2014_train_0006')
    records = dict(img_id=1, ann_id=1, num_no_objects=0, vid_train_frames=0)
    obj_num_classes = dict()

    vid_train_img_list = osp.join(ann_dir, 'Lists/DET_train_30classes.txt')
    vid_train_img_list = mmcv.list_from_file(vid_train_img_list)
    vid_train_img_names = []
    for vid_train_img_info in vid_train_img_list:
        vid_train_img_names.append(f"{vid_train_img_info.split(' ')[0]}.JPEG")

    for img_name in tqdm(vid_train_img_names):
        xml_path = osp.join(ann_dir, 'Annotations/DET',
                            img_name.replace('JPEG', 'xml'))
        records, DET, obj_num_classes = parse_xml(img_name, xml_path, True,
                                                  records, DET,
                                                  obj_num_classes)

    for sub_set in tqdm(dataset_sets):
        sub_set_base_path = osp.join(ann_dir, 'Annotations/DET', sub_set)
        if 'train/ILSVRC2013_train' == sub_set:
            xml_paths = sorted(
                glob.glob(osp.join(sub_set_base_path, '*', '*.xml')))
        else:
            xml_paths = sorted(glob.glob(osp.join(sub_set_base_path, '*.xml')))

        for xml_path in tqdm(xml_paths):
            img_name = xml_path.replace(sub_set_base_path, sub_set)
            img_name = img_name.replace('xml', 'JPEG')
            is_vid_train_frame = False
            if img_name in vid_train_img_names:
                continue

            records, DET, obj_num_classes = parse_xml(img_name, xml_path,
                                                      is_vid_train_frame,
                                                      records, DET,
                                                      obj_num_classes)
    if not osp.isdir(save_dir):
        os.makedirs(save_dir)
    mmcv.dump(DET, osp.join(save_dir, 'imagenet_det_30plus1cls.json'))
    print('-----ImageNet DET------')
    print(f'total {records["img_id"] - 1} images')
    print(f'{records["num_no_objects"]} images have no objects')
    print(f'total {records["vid_train_frames"]} images '
          'for video detection training')
    print(f'{records["ann_id"] - 1} objects are annotated.')
    print('-----------------------')
    for i in range(1, len(CLASSES) + 1):
        print(f'Class {i} {CLASSES[i - 1]} has {obj_num_classes[i]} objects.')


def main():
    args = parse_args()

    DET = defaultdict(list)
    for k, v in enumerate(CLASSES, 1):
        if k == len(CLASSES):
            DET['categories'].append(dict(id=k, name=v, encode_name=None))
        else:
            DET['categories'].append(
                dict(id=k, name=v, encode_name=CLASSES_ENCODES[k - 1]))
    convert_det(DET, args.input, args.output)


if __name__ == '__main__':
    main()
