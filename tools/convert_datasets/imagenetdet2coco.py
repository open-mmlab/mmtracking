import argparse
import os.path as osp
import xml.etree.ElementTree as ET
from collections import defaultdict

import mmcv
from tqdm import tqdm


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


def convert_det(DET, ann_dir, save_dir):
    records = dict(ann_id=0, num_no_objects=0)
    obj_num_classes = dict()
    img_list = osp.join(ann_dir, 'Lists/DET_train_30classes.txt')
    xml_dir = osp.join(ann_dir, 'Annotations/DET/')
    img_list = mmcv.list_from_file(img_list)
    # random.shuffle(img_list)
    for img_id, img_info in tqdm(enumerate(img_list)):
        img_info = img_info.split(' ')
        xml_name = osp.join(xml_dir, '{}.xml'.format(img_info[0]))
        # parse XML annotation file
        tree = ET.parse(xml_name)
        root = tree.getroot()
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        image = dict(
            file_name='{}.JPEG'.format(img_info[0]),
            height=height,
            width=width,
            id=img_id)
        DET['images'].append(image)
        if root.findall('object') == []:
            print(xml_name, 'has no objects.')
            records['num_no_objects'] += 1
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
            ann = dict(
                id=records['ann_id'],
                image_id=img_id,
                category_id=category_id,
                bbox=[x1, y1, w, h],
                area=w * h,
                iscrowd=False,
                ignore=False,
                segmentation=[[x1, y1, x1, y2, x2, y2, x2, y1]])
            DET['annotations'].append(ann)
            records['ann_id'] += 1
            if category_id not in obj_num_classes.keys():
                obj_num_classes[category_id] = 1
            else:
                obj_num_classes[category_id] += 1
    mmcv.dump(DET, osp.join(save_dir, 'ImageNet_DET_30cls.json'))
    print('-----ImageNet DET------')
    print('{} images for training'.format(img_id + 1))
    print('{} images have no objects'.format(records['num_no_objects']))
    print('{} objects are annotated.'.format(records['ann_id']))
    print('-----------------------')
    # for k, v in obj_num_classes.items():
    #     print('Class {} {} has {} objects.'.format(k, CLASSES[k - 1], v))
    for i in range(1, 31):
        print('Class {} {} has {} objects.'.format(i, CLASSES[i - 1],
                                                   obj_num_classes[i]))


def main():
    args = parse_args()

    DET = defaultdict(list)
    for k, v in enumerate(CLASSES, 1):
        DET['categories'].append(dict(id=k, name=v))
    DET = convert_det(DET, args.ann_dir, args.save_dir)


if __name__ == '__main__':
    main()
    # srun -p ad_rs python -u imagenetdet2coco.py -d ../data/ILSVRC -s .
