import argparse
import os
from collections import defaultdict

import mmcv

cats_mapping = {'pedestrain': 1}
'''MOT Annotations
    gt.txt
    ----------
    0, frame_id
    1, instance_id
    2, x1
    3, y1
    4, w
    5, h
    6, conf       [1.0] 0.0 is ignored
    7, cat_id
    8, visibility [0.25]

    classes
    ----------
    1: [PERSON] 'pedestrian'
    2: [IGNORE] 'person on vehicle'
    3: [USELESS] 'car'
    4: [USELESS] 'bicycle'
    5: [USELESS] 'motorbike'
    6: [USELESS] 'non motorized vehicle'
    7: [IGNORE] 'static person'
    8: [IGNORE] 'distractor'
    9: [USELESS] 'occluder'
    10: [USELESS] 'occluder on the ground',
    11: [USELESS] 'occluder full'
    12: [IGNORE] 'reflection'
'''


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MOT label and detections to COCO-VID format.')
    parser.add_argument('-i', '--input', help='path of MOT data')
    parser.add_argument(
        '-o', '--output', help='path to save coco formatted label file')
    return parser.parse_args()


def main():
    args = parse_args()
    vid_folders = os.listdir(args.ann_dir)
    outputs = defaultdict(list)
    img_id, ann_id, global_ins_id = 0, 0, 0
    for k, v in cats_mapping.items():
        outputs['categories'].append(dict(id=v, name=k))
    include_cls = [1]

    for vid_id, vid_name in enumerate(vid_folders):
        max_frame = -1
        max_id = -1
        txt_name = '{}/{}/gt/gt.txt'.format(args.ann_dir, vid_name)
        inf_name = '{}/{}/seqinfo.ini'.format(args.ann_dir, vid_name)
        with open(inf_name, 'r') as f:
            infs = f.readlines()
            width = int(infs[5].strip().split('=')[1])
            height = int(infs[6].strip().split('=')[1])
            print("Video {}: width {}, height {}".format(
                vid_name, width, height))
        # for vid
        vid_info = dict(id=vid_id, name=vid_name)
        outputs['videos'].append(vid_info)
        img_id_pool = []
        with open(txt_name, 'r') as f:
            # for image & anns
            vid_anns = f.readlines()
            for vid_ann in vid_anns:
                # parse the line
                frame, id, x1, y1, w, h, ignore, c, vis = map(
                    float,
                    vid_ann.strip().split(','))
                frame = int(frame)
                id = int(id)
                ignore = int(ignore)
                c = int(c)
                if id > max_id:
                    max_id = id
                if frame > max_frame:
                    max_frame = frame
                cur_img_id = img_id + frame
                cur_ins_id = global_ins_id + id
                if not c in include_cls:
                    continue
                if cur_img_id not in img_id_pool:
                    img_id_pool.append(cur_img_id)
                    img_info = dict(
                        file_name='{}/img1/{}.jpg'.format(
                            vid_name,
                            str(frame).zfill(6)),
                        height=height,
                        width=width,
                        id=cur_img_id,
                        video_id=vid_id,
                        index=frame - 1)
                    print(frame - 1)
                    outputs['images'].append(img_info)
                    # print("Image id: {}".format(cur_img_id))
                if vis > 0.1:
                    ann = dict(
                        id=ann_id,
                        image_id=cur_img_id,
                        category_id=1,
                        instance_id=cur_ins_id,
                        is_occluded=False,
                        is_truncated=False,
                        bbox=[x1, y1, w, h],
                        area=w * h,
                        iscrowd=0,
                        ignore=0,
                        segmentation=[[0, 0, 0, 0, 0, 0, 0, 0]])
                    outputs['annotations'].append(ann)
                    ann_id += 1
        img_id += max_frame
        global_ins_id += max_id
    mmcv.dump(outputs, '{}/mot17_train.json'.format(args.save_dir))


if __name__ == "__main__":
    main()
