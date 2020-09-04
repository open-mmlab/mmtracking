import argparse
from collections import defaultdict

import mmcv

from mmtrack.datasets import CocoVID


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert COCO-style video annotations \
            to OpenMMLab format')
    parser.add_argument('-i', '--input', help='input file')
    parser.add_argument('-o', '--output', help='output file')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    vid = CocoVID(args.input)
    mmvid = defaultdict(list)

    # classes
    cat_ids = vid.getCatIds()
    cat2label = dict()
    for i, cat_id in enumerate(cat_ids):
        cat2label[cat_id] = i
        cat_info = vid.loadCats(cat_id)[0]
        mmvid['classes'].append(cat_info['name'])

    # data
    vid_ids = vid.getVidIds()
    for vid_id in vid_ids:
        mm_vid_info = defaultdict(list)
        vid_info = vid.loadVids(vid_id)[0]
        img_ids = vid.getImgIdsFromVidId(vid_id)
        if len(img_ids) == 0:
            raise ValueError('No images in the video.')
        else:
            for img_id in img_ids:
                img_info = vid.loadImgs(img_id)[0]
                mm_img_info = defaultdict(list)
                ann_ids = vid.getAnnIds(img_id)
                if len(ann_ids) == 0:
                    mm_img_info['annotations'] = []
                else:
                    for ann_id in ann_ids:
                        ann = vid.loadAnns(ann_id)[0]
                        x1, y1, w, h = ann['bbox']
                        mmann = dict(
                            bbox=[x1, y1, x1 + w, y1 + h],
                            label=cat2label[ann['category_id']],
                            instance_id=ann['instance_id'],
                            # ignore=ann['ignore'],
                            crowd=ann['iscrowd'],
                            occluded=ann['occluded'],
                            truncated=ann['truncated'])
                        mm_img_info['annotations'].append(mmann)
                mm_img_info['name'] = img_info['file_name']
                mm_img_info['frame_id'] = img_info['frame_id']
                mm_vid_info['images'].append(mm_img_info)
        mm_vid_info['width'] = img_info['width']
        mm_vid_info['height'] = img_info['height']
        mm_vid_info['length'] = len(mm_vid_info['images'])
        mm_vid_info['fps'] = 5
        mm_vid_info['name'] = vid_info['name']
        mmvid['data'].append(mm_vid_info)

    # license
    mmvid['metas'] = dict()

    # save
    mmcv.dump(mmvid, args.output)
    print('Done!')


if __name__ == '__main__':
    main()
