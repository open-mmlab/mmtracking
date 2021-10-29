# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from collections import defaultdict

import cv2
import mmcv
import numpy as np


def create_dummy_data():
    vot_test = defaultdict(list)
    records = dict(vid_id=1, img_id=1, ann_id=1, global_instance_id=1)
    videos_list = ['drone_across', 'matrix']
    imgs_size = dict(drone_across=(720, 1280), matrix=(336, 800))

    vot_test['categories'] = [dict(id=0, name=0)]

    for video_name in videos_list:
        video_path = video_name
        video = dict(id=records['vid_id'], name=video_name)
        vot_test['videos'].append(video)

        gt_bboxes = mmcv.list_from_file(
            osp.join(video_path, 'groundtruth.txt'))

        height, width = imgs_size[video_name]
        for frame_id, gt_bbox in enumerate(gt_bboxes):
            file_name = '%08d' % (frame_id + 1) + '.jpg'
            file_name = osp.join(video_name, 'color', file_name)
            image = dict(
                file_name=file_name,
                height=height,
                width=width,
                id=records['img_id'],
                frame_id=frame_id,
                video_id=records['vid_id'])
            vot_test['images'].append(image)

            bbox = gt_bbox.split(',')
            bbox = list(map(lambda x: int(float(x)), bbox))
            if len(bbox) == 4:
                bbox = [
                    bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1],
                    bbox[0] + bbox[2], bbox[1] + bbox[3], bbox[0],
                    bbox[1] + bbox[3]
                ]
            ann = dict(
                id=records['ann_id'],
                image_id=records['img_id'],
                instance_id=records['global_instance_id'],
                category_id=0,
                bbox=bbox,
                area=cv2.contourArea(np.array(bbox).reshape(4, 2)))
            vot_test['annotations'].append(ann)

            records['ann_id'] += 1
            records['img_id'] += 1
        records['global_instance_id'] += 1
        records['vid_id'] += 1

    mmcv.dump(vot_test, 'vot_test_dummy.json')


if __name__ == '__main__':
    create_dummy_data()
