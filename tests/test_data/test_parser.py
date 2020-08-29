import sys

import numpy as np

from mmtrack.datasets import mmVID


def test_mmvid_parser(ann_file):
    mmvid = mmVID(ann_file)

    # cls2label
    labels = mmvid.cls2label(['car', 'truck'])
    assert isinstance(labels, list)
    assert labels[0] == 2 and labels[1] == 4
    label = mmvid.cls2label('car')
    assert isinstance(label, int)
    assert label == 2

    # vid_info
    vid_id = np.random.randint(len(mmvid.videos))
    vid_info = mmvid.get_vid(vid_id)
    assert isinstance(vid_info['name'], str)
    assert isinstance(vid_info['width'], int)
    assert isinstance(vid_info['height'], int)
    assert isinstance(vid_info['length'], int)
    assert isinstance(vid_info['fps'], int)
    assert 'images' in vid_info.keys()
    frame_ids = [_['frame_id'] for _ in vid_info['images']]
    _frame_ids = sorted(frame_ids)
    assert frame_ids == _frame_ids

    # img_info
    img_id = np.random.randint(len(mmvid.images))
    img_info = mmvid.get_img(img_id)
    assert isinstance(img_info['name'], str)
    assert isinstance(img_info['frame_id'], int)
    assert 'annotations' in img_info.keys()

    # ann_info
    anns = mmvid.get_anns(img_id, parse=True)
    print('{} objects in all classes.'.format(len(anns['labels'])))
    print(anns.keys())
    for k, v in anns.items():
        print('{}: {}'.format(k, v))
    anns = mmvid.get_anns(img_id, classes=['car', 'truck'])
    print('{} objects in car and truck.'.format(len(anns['labels'])))

    car_img_infos = mmvid.get_imgs_by_cls('car')
    print('{} images have car.'.format(len(car_img_infos)))

    all_ins_ids = mmvid.get_ins_ids()
    print('{} instances in the dataset.'.format(len(all_ins_ids)))
    ins_ids = mmvid.get_ins_ids(vid_id)
    print('{} instances in video {}.'.format(len(ins_ids), vid_id))

    front_imgs, behind_imgs = mmvid.get_neighbor_imgs(img_id, scope=3)
    for _ in front_imgs + behind_imgs:
        print(_['frame_id'])


if __name__ == '__main__':
    test_mmvid_parser(sys.argv[1])
