import argparse
import os.path as osp
from collections import defaultdict

import mmcv
from tao.toolkit.tao import Tao
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='Make annotation files for TAO')
    parser.add_argument('-t', '--tao', help='path of TAO json file')
    parser.add_argument(
        '--filter-classes',
        action='store_true',
        help='whether filter 1230 classes to 482.')
    return parser.parse_args()


def get_classes(tao_path, filter_classes=True):
    train = mmcv.load(osp.join(tao_path, 'train.json'))

    train_classes = list(set([_['category_id'] for _ in train['annotations']]))
    print(f'TAO train set contains {len(train_classes)} categories.')

    val = mmcv.load(osp.join(tao_path, 'validation.json'))
    val_classes = list(set([_['category_id'] for _ in val['annotations']]))
    print(f'TAO val set contains {len(val_classes)} categories.')

    test = mmcv.load(osp.join(tao_path, 'test_categories.json'))
    test_classes = list(set([_['id'] for _ in test['categories']]))
    print(f'TAO test set contains {len(test_classes)} categories.')

    tao_classes = set(train_classes + val_classes + test_classes)
    print(f'TAO totally contains {len(tao_classes)} categories.')

    tao_classes = [_ for _ in train['categories'] if _['id'] in tao_classes]

    with open(osp.join(tao_path, 'tao_classes.txt'), 'wt') as f:
        for c in tao_classes:
            name = c['name']
            f.writelines(f'{name}\n')

    if filter_classes:
        return tao_classes
    else:
        return train['categories']


def convert_tao(file, classes):
    tao = Tao(file)
    raw = mmcv.load(file)

    out = defaultdict(list)
    out['tracks'] = raw['tracks'].copy()
    out['info'] = raw['info'].copy()
    out['licenses'] = raw['licenses'].copy()
    out['categories'] = classes

    for video in tqdm(raw['videos']):
        img_infos = tao.vid_img_map[video['id']]
        img_infos = sorted(img_infos, key=lambda x: x['frame_index'])
        frame_range = img_infos[1]['frame_index'] - img_infos[0]['frame_index']
        video['frame_range'] = frame_range
        out['videos'].append(video)
        for i, img_info in enumerate(img_infos):
            img_info['frame_id'] = i
            img_info['neg_category_ids'] = video['neg_category_ids']
            img_info['not_exhaustive_category_ids'] = video[
                'not_exhaustive_category_ids']
            out['images'].append(img_info)
            ann_infos = tao.img_ann_map[img_info['id']]
            for ann_info in ann_infos:
                ann_info['instance_id'] = ann_info['track_id']
                out['annotations'].append(ann_info)

    assert len(out['videos']) == len(raw['videos'])
    assert len(out['images']) == len(raw['images'])
    assert len(out['annotations']) == len(raw['annotations'])
    return out


def main():
    args = parse_args()

    classes = get_classes(args.tao, args.filter_classes)
    print(f'convert with {len(classes)} classes')

    for file in [
            'train.json', 'validation.json', 'test_without_annotations.json'
    ]:
        print(f'convert {file}')
        out = convert_tao(osp.join(args.tao, file), classes)
        c = '_482' if args.filter_classes else ''
        prefix = file.split('.')[0].split('_')[0]
        out_file = f'{prefix}{c}_ours.json'
        mmcv.dump(out, osp.join(args.tao, out_file))


if __name__ == '__main__':
    main()
