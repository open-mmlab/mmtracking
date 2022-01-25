# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from collections import defaultdict

import mmcv


def create_dummy_data():
    lasot_test = defaultdict(list)
    records = dict(vid_id=1, img_id=1, ann_id=1, global_instance_id=1)
    videos_list = ['airplane-1', 'airplane-2']

    lasot_test['categories'] = [dict(id=0, name=0)]

    for video_name in videos_list:
        video_path = video_name
        video = dict(id=records['vid_id'], name=video_name)
        lasot_test['videos'].append(video)

        gt_bboxes = mmcv.list_from_file(
            osp.join(video_path, 'groundtruth.txt'))

        height, width, _ = (360, 640, 3)
        for frame_id, gt_bbox in enumerate(gt_bboxes):
            file_name = '%08d' % (frame_id + 1) + '.jpg'
            file_name = osp.join(video_name, 'img', file_name)
            image = dict(
                file_name=file_name,
                height=height,
                width=width,
                id=records['img_id'],
                frame_id=frame_id,
                video_id=records['vid_id'])
            lasot_test['images'].append(image)

            x1, y1, w, h = gt_bbox.split(',')
            ann = dict(
                id=records['ann_id'],
                image_id=records['img_id'],
                instance_id=records['global_instance_id'],
                category_id=0,
                bbox=[int(x1), int(y1), int(w),
                      int(h)],
                area=int(w) * int(h),
                full_occlusion=False,
                out_of_view=False)
            lasot_test['annotations'].append(ann)

            records['ann_id'] += 1
            records['img_id'] += 1
        records['global_instance_id'] += 1
        records['vid_id'] += 1

    mmcv.dump(lasot_test, 'lasot_test_dummy.json')


if __name__ == '__main__':
    create_dummy_data()
