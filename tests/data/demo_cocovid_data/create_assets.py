# Copyright (c) OpenMMLab. All rights reserved.
from collections import defaultdict

import mmcv
from mmdet.core.bbox.demodata import random_boxes


def create_dummy_data():
    # 2 categories: ['car', 'person']
    classes = ['car', 'person']
    # 1 video
    videos = [
        dict(
            name='dummy_video',
            images=dict(num=8, shape=(256, 512, 3)),
            instances=[
                dict(frames=[1, 2, 3, 4], category='car'),
                dict(frames=[4, 5, 6], category='person')
            ])
    ]
    attrs = dict(occluded=False, truncated=False, iscrowd=False, ignore=False)
    attrs['is_vid_train_frame'] = True  # ImageNet VID
    attrs['visibility'] = 1.0  # MOT17
    # set all corner cases in img_id == 1
    corner_cases = dict(ignore=0, iscrowd=3)

    ann = defaultdict(list)
    img_id, ann_id, ins_id = 1, 1, 1
    for cls_id, cls in enumerate(classes, 1):
        ann['categories'].append(dict(id=cls_id, name=cls))

    for vid_id, video in enumerate(videos, 1):
        ann['videos'].append(dict(id=vid_id, name=video['name']))

        img_info = video['images']
        frame2id = dict()
        for i in range(img_info['num']):
            img_name = f'image_{img_id}.jpg'
            # img = np.ones(img_info['shape']) * 125
            # mmcv.imwrite(img, img_name)
            ann['images'].append(
                dict(
                    file_name=img_name,
                    height=img_info['shape'][0],
                    width=img_info['shape'][1],
                    id=img_id,
                    video_id=vid_id,
                    frame_id=i))
            frame2id[i] = img_id
            img_id += 1

        ins_info = video['instances']
        for i, ins in enumerate(ins_info):
            bboxes = random_boxes(
                len(ins['frames']), min(img_info['shape'][:-1])).numpy()
            for ind, frame_id in enumerate(ins['frames']):
                assert frame_id < img_info['num']
                x1 = float(bboxes[ind][0])
                y1 = float(bboxes[ind][1])
                x2 = float(bboxes[ind][2])
                y2 = float(bboxes[ind][3])
                bbox = [x1, y1, x2 - x1, y2 - y1]
                area = float((x2 - x1) * (y2 - y1))
                bbox[2] = 2.0 if bbox[2] < 1 else bbox[2]
                bbox[3] = 2.0 if bbox[2] < 1 else bbox[3]
                ann['annotations'].append(
                    dict(
                        id=ann_id,
                        image_id=frame2id[frame_id],
                        video_id=vid_id,
                        category_id=classes.index(ins['category']) + 1,
                        instance_id=ins_id,
                        bbox=bbox,
                        area=area,
                        **attrs))
                ann_id += 1
            ins_id += 1

    for case, num in corner_cases.items():
        bboxes = random_boxes(num, min(img_info['shape'][:-1]) - 1).numpy()
        for ind in range(bboxes.shape[0]):
            x1 = float(bboxes[ind][0])
            y1 = float(bboxes[ind][1])
            x2 = float(bboxes[ind][2])
            y2 = float(bboxes[ind][3])
            bbox = [x1, y1, x2 - x1, y2 - y1]
            bbox[2] = 2.0 if bbox[2] < 1 else bbox[2]
            bbox[3] = 2.0 if bbox[3] < 1 else bbox[3]
            area = float((x2 - x1) * (y2 - y1))
            _attrs = attrs.copy()
            if case == 'ignore':
                _attrs['ignore'] = True
            elif case == 'iscrowd':
                _attrs['iscrowd'] = True
            elif case == 'visibility':
                _attrs['visibility'] = 1.0
            else:
                raise KeyError()
            ann['annotations'].append(
                dict(
                    id=ann_id,
                    image_id=1,
                    video_id=1,
                    category_id=1,
                    instance_id=ins_id,
                    bbox=bbox,
                    area=area,
                    **_attrs))
            ann_id += 1
            ins_id += 1

    mmcv.dump(ann, 'ann.json')


if __name__ == '__main__':
    create_dummy_data()
