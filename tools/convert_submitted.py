import json
import os
import os.path as osp
from collections import defaultdict


def dump_trackingnet_results(ann_file, track_results, save_dir):
    with open(ann_file, 'r') as f:
        info = json.load(f)
        video_info = info['videos']
        imgs_info = info['images']

    print('-------- Image Number: {} --------'.format(len(track_results)))

    new_results = defaultdict(list)
    for img_id, bbox in enumerate(track_results):
        img_info = imgs_info[img_id]
        assert img_info['id'] == img_id + 1, 'img id is not matched'
        video_name = img_info['file_name'].split('/')[0]
        new_results[video_name].append(track_results[img_id][:4])

    assert len(video_info) == len(
        new_results), 'video number is not right {}--{}'.format(
            len(video_info), len(new_results))

    if not osp.isdir(save_dir):
        os.makedirs(save_dir)
    for v_name, bboxes in new_results.items():
        vid_txt = osp.join(save_dir, '{}.txt'.format(v_name))
        with open(vid_txt, 'w') as f:
            for i, bbox in enumerate(bboxes):
                bbox = [
                    str(bbox[0]),
                    str(bbox[1]),
                    str(bbox[2] - bbox[0]),
                    str(bbox[3] - bbox[1])
                ]
                if i == 0:
                    line = ','.join(bbox)
                else:
                    line = '\n' + ','.join(bbox)
                f.writelines(line)
    print('writing submitted results to {}'.format(save_dir))
