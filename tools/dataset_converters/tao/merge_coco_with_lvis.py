# Copyright (c) OpenMMLab. Modofied from script in TAO-Dataset/tao https://github.com/TAO-Dataset/tao.git # noqa
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pycocotools.mask as mask_util
from pycocotools.coco import COCO
from tqdm import tqdm


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--lvis', type=Path, required=True, help='lvis json path')
    parser.add_argument(
        '--coco', type=Path, required=True, help='coco json path')
    parser.add_argument(
        '--mapping',
        type=Path,
        required=True,
        help='synset mapping from coco to lvis')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument(
        '--iou-thresh',
        default=0.7,
        type=float,
        help=('If a COCO annotation overlaps with an LVIS annotations with '
              'IoU over this threshold, we use only the LVIS annotation.'))

    args = parser.parse_args()
    args.output.parent.mkdir(exist_ok=True, parents=True)

    coco = COCO(args.coco)
    lvis = COCO(args.lvis)

    # transfer COCO category name LVIS according to synset
    # synset format
    # "bench": {
    # "coco_cat_id": 15,
    # "meaning": "a long seat for more than one person",
    # "synset": "bench.n.01"}
    synset2lvis = {cat['syn set']: cat['id'] for cat in lvis.cats.values()}
    coco2lvis = {}
    with open(args.mapping, 'r') as f:
        mapping = json.load(f)
    for cat in coco.cats.values():
        mapped = mapping[cat['name']]
        assert mapped['coco_cat_id'] == cat['id']
        synset = mapped['synset']
        if synset not in synset2lvis:
            print(f'Found no LVIS category for "{cat["name"]}" from COCO')
            continue
        coco2lvis[cat['id']] = synset2lvis[synset]

    for img_id, _ in coco.imgs.items():
        if img_id in lvis.imgs:
            coco_name = coco.imgs[img_id]['file_name']
            lvis_name = lvis.imgs[img_id]['file_name']
            assert coco_name in lvis_name
        else:
            print(f'Image {img_id} in COCO, but not annotated in LVIS')

    # add coco annotations at the end of lvis's
    lvis_highest_id = max(x['id'] for x in lvis.anns.values())
    ann_id_generator = itertools.count(lvis_highest_id + 1)
    new_annotations = []
    for img_id, lvis_anns in tqdm(lvis.imgToAnns.items()):
        if img_id not in coco.imgToAnns:
            print(f'Image {img_id} in LVIS, but not annotated in COCO')
            continue

        coco_anns = coco.imgToAnns[img_id]
        # Compute IoU between coco_anns and lvis_anns
        # Shape (num_coco_anns, num_lvis_anns)
        mask_iou = mask_util.iou([coco.annToRLE(x) for x in coco_anns],
                                 [lvis.annToRLE(x) for x in lvis_anns],
                                 pyiscrowd=np.zeros(len(lvis_anns)))
        does_overlap = mask_iou.max(axis=1) > args.iou_thresh
        to_add = []
        for i, ann in enumerate(coco_anns):
            if does_overlap[i]:
                continue
            if ann['category_id'] not in coco2lvis:
                continue
            ann['category_id'] = coco2lvis[ann['category_id']]
            ann['id'] = next(ann_id_generator)
            to_add.append(ann)
        new_annotations.extend(to_add)

    with open(args.lvis, 'r') as f:
        merged = json.load(f)
    merged['annotations'].extend(new_annotations)
    with open(args.output, 'w') as f:
        json.dump(merged, f)


if __name__ == '__main__':
    main()
