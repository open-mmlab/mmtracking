import argparse
import os.path as osp
from collections import defaultdict

import mmcv
from tqdm import tqdm

CLASSES = [
    'pedestrian', 'rider', 'car', 'bus', 'truck', 'bicycle', 'motorcycle',
    'train'
]
USELESS = ['traffic light', 'traffic sign']
IGNORES = ['trailer', 'other person', 'other vehicle']


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert BDD100K detection label to COCO format')
    parser.add_argument('-i', '--input', help='path of BDD label json file')
    parser.add_argument(
        '-o', '--output', help='path to save coco formatted label file')
    return parser.parse_args()


def main():
    args = parse_args()

    for subset in ['train', 'val']:
        print(f'convert BDD100K detection {subset} set into coco format')

        bdd = mmcv.load(osp.join(args.input, f'det_v2_{subset}_release.json'))
        coco = defaultdict(list)

        for cls_id, cls in enumerate(CLASSES, 1):
            coco['categories'].append(dict(id=cls_id, name=cls))

        ann_id = 0
        for img_id, img_info in enumerate(tqdm(bdd)):
            if img_info['labels'] is None:
                continue
            img = dict(
                file_name=img_info['name'],
                height=720,
                width=1280,
                id=img_id,
                metas=img_info['attributes'])
            coco['images'].append(img)
            for k, ann_info in enumerate(img_info['labels']):
                if ann_info['category'] in CLASSES:
                    cls_id = CLASSES.index(ann_info['category']) + 1
                elif ann_info['category'] in USELESS or ann_info[
                        'category'] in IGNORES:
                    continue
                else:
                    raise ValueError('Category do not exist.')
                x1 = ann_info['box2d']['x1']
                x2 = ann_info['box2d']['x2']
                y1 = ann_info['box2d']['y1']
                y2 = ann_info['box2d']['y2']
                area = float((x2 - x1) * (y2 - y1))
                ann = dict(
                    id=ann_id,
                    image_id=img_id,
                    category_id=cls_id,
                    bbox=[x1, y1, x2 - x1, y2 - y1],
                    area=area,
                    occluded=ann_info['attributes']['occluded'],
                    truncated=ann_info['attributes']['truncated'])
                coco['annotations'].append(ann)
                ann_id += 1
        mmcv.dump(coco, osp.join(args.output, f'{subset}_coco-format.json'))
        print('converted {} images with {} objects'.format(
            len(coco['images']), len(coco['annotations'])))


if __name__ == '__main__':
    main()
