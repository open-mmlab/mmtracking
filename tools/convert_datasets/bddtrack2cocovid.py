import argparse
import os
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
        description='Convert BDD100K tracking label to COCO-VID format')
    parser.add_argument('-i', '--input', help='path of BDD label json file')
    parser.add_argument(
        '-o', '--output', help='path to save coco formatted label file')
    return parser.parse_args()


def main():
    args = parse_args()

    for subset in ['train', 'val']:
        print(f'convert BDD tracking {subset} set into COCO-VID format')
        vids = os.listdir(osp.join(args.input, subset))
        coco = defaultdict(list)

        for cls_id, cls in enumerate(CLASSES, 1):
            coco['categories'].append(dict(id=cls_id, name=cls))

        img_id, ann_id = 0, 0
        for vid_id, vid_name in enumerate(tqdm(vids)):
            vid_infos = mmcv.load(osp.join(args.input, subset, vid_name))
            video = dict(id=vid_id, name=vid_infos[0]['video_name'])
            coco['videos'].append(video)
            for img_info in vid_infos:
                image = dict(
                    file_name=osp.join(img_info['video_name'],
                                       img_info['name']),
                    height=720,
                    width=1280,
                    id=img_id,
                    video_id=vid_id,
                    frame_id=img_info['index'])
                coco['images'].append(image)
                for ann_info in img_info['labels']:
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
                        instance_id=ann_info['id'],
                        bbox=[x1, y1, x2 - x1, y2 - y1],
                        area=area,
                        occluded=ann_info['attributes']['Occluded'],
                        truncated=ann_info['attributes']['Truncated'],
                        iscrowd=ann_info['attributes']['Crowd'])
                    coco['annotations'].append(ann)
                    ann_id += 1
                img_id += 1
        mmcv.dump(
            coco,
            osp.join(args.output, f'bdd100k_track_{subset}_cocoformat.json'))
        print('converted {} videos, {} images with {} objects'.format(
            len(coco['videos']), len(coco['images']),
            len(coco['annotations'])))


if __name__ == '__main__':
    main()
