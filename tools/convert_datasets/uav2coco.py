import argparse
import os
import os.path as osp
from collections import defaultdict

import mmcv
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='UAV123 dataset to COCO Video format')
    parser.add_argument(
        '-i',
        '--input',
        help='root directory of UAV123 dataset',
    )
    parser.add_argument(
        '-o',
        '--output',
        help='directory to save coco formatted label file',
    )
    return parser.parse_args()


def convert_uav123(uav123, ann_dir, save_dir):
    """Convert trackingnet dataset to COCO style.

    Args:
        uav123 (dict): The converted COCO style annotations.
        ann_dir (str): The path of trackingnet test dataset
        save_dir (str): The path to save `uav123`.
    """
    # The format of each line in "uav_info.txt" is
    # "anno_name,anno_path,img_path,start_frame,end_frame"
    info_path = osp.join(os.path.dirname(__file__), 'uav_info.txt')
    uav_info = mmcv.list_from_file(info_path)

    records = dict(vid_id=1, img_id=1, ann_id=1, global_instance_id=1)
    uav123['categories'] = [dict(id=0, name=0)]

    for info in tqdm(uav_info):
        anno_name, anno_path, video_path, start_frame, end_frame = info.split(
            ',')
        start_frame = int(start_frame)
        end_frame = int(end_frame)

        # video_name is not the same as anno_name since one video may have
        # several fragments.
        # Example: video_name: "bird"   anno_name: "bird_1"
        video_name = video_path.split('/')[-1]
        video = dict(id=records['vid_id'], name=video_name)
        uav123['videos'].append(video)

        gt_bboxes = mmcv.list_from_file(osp.join(ann_dir, anno_path))
        assert len(gt_bboxes) == end_frame - start_frame + 1

        img = mmcv.imread(
            osp.join(ann_dir, video_path, '%06d.jpg' % (start_frame)))
        height, width, _ = img.shape
        for frame_id in range(start_frame, end_frame + 1):
            file_name = osp.join(video_name, '%06d.jpg' % (frame_id))
            bbox_id = frame_id - start_frame
            image = dict(
                file_name=file_name,
                height=height,
                width=width,
                id=records['img_id'],
                frame_id=bbox_id,
                video_id=records['vid_id'])
            uav123['images'].append(image)

            if 'NaN' in gt_bboxes[bbox_id]:
                x1 = y1 = w = h = 0
                ignore = True
            else:
                x1, y1, w, h = gt_bboxes[bbox_id].split(',')
                ignore = False
            ann = dict(
                id=records['ann_id'],
                image_id=records['img_id'],
                instance_id=records['global_instance_id'],
                category_id=0,
                bbox=[int(x1), int(y1), int(w),
                      int(h)],
                area=int(w) * int(h),
                ignore=ignore)
            uav123['annotations'].append(ann)

            records['ann_id'] += 1
            records['img_id'] += 1

        records['global_instance_id'] += 1
        records['vid_id'] += 1

    mmcv.dump(uav123, osp.join(save_dir, 'uav123.json'))
    print('-----UAV123 Dataset------')
    print(f'{records["vid_id"]- 1} videos')
    print(f'{records["global_instance_id"]- 1} instances')
    print(f'{records["img_id"]- 1} images')
    print(f'{records["ann_id"] - 1} objects')
    print('-----------------------------')


def dataset_check(ann_dir):
    print('----- Check Dataset Json File -----')
    with open(osp.join(ann_dir, 'uav123.json'), 'r') as f:
        anno = mmcv.load(f, file_format='json')
    print('annotation keys: ', list(anno.keys()))
    for key, value in anno.items():
        print(key, value[0])
        if len(value) > 1:
            print(key, value[1])
    print('-----------------------------')


def main():
    args = parse_args()
    uav123 = defaultdict(list)
    convert_uav123(uav123, args.input, args.output)
    dataset_check(args.output)


if __name__ == '__main__':
    main()
