# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

import mmcv
import motmetrics as mm
import numpy as np
from mmcv import Config
from mmcv.utils import print_log

from mmtrack.core.utils import imshow_mot_errors
from mmtrack.datasets import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='visualize errors for multiple object tracking')
    parser.add_argument('config', help='path of the config file')
    parser.add_argument('--result-file', help='path of the inference result')
    parser.add_argument(
        '--out-dir',
        help='directory where painted images or videos will be saved')
    parser.add_argument(
        '--show',
        action='store_true',
        help='whether to show the results on the fly')
    parser.add_argument(
        '--fps', type=int, default=3, help='FPS of the output video')
    parser.add_argument(
        '--backend',
        type=str,
        choices=['cv2', 'plt'],
        default='cv2',
        help='backend of visualization')
    args = parser.parse_args()
    return args


def compare_res_gts(resfiles, dataset, video_name):
    """Evaluate the results of the video.

    Args:
        resfiles (dict): A dict containing the directory of the MOT results.
        dataset (Dataset): MOT dataset of the video to be evaluated.
        video_name (str): Name of the video to be evaluated.

    Returns:
        tuple: (acc, res, gt), acc contains the results of MOT metrics,
        res is the results of inference and gt is the ground truth.
    """
    if 'half-train' in dataset.ann_file:
        gt_file = osp.join(dataset.img_prefix,
                           f'{video_name}/gt/gt_half-train.txt')
    elif 'half-val' in dataset.ann_file:
        gt_file = osp.join(dataset.img_prefix,
                           f'{video_name}/gt/gt_half-val.txt')
    else:
        gt_file = osp.join(dataset.img_prefix, f'{video_name}/gt/gt.txt')
    res_file = osp.join(resfiles['track'], f'{video_name}.txt')
    gt = mm.io.loadtxt(gt_file)
    res = mm.io.loadtxt(res_file)
    ini_file = osp.join(dataset.img_prefix, f'{video_name}/seqinfo.ini')
    if osp.exists(ini_file):
        acc, ana = mm.utils.CLEAR_MOT_M(gt, res, ini_file)
    else:
        acc = mm.utils.compare_to_groundtruth(gt, res)

    return acc, res, gt


def main():
    args = parse_args()

    assert args.show or args.out_dir, \
        ('Please specify at least one operation (show the results '
         '/ save the results) with the argument "--show" or "--out-dir"')

    if not args.result_file.endswith(('.pkl', 'pickle')):
        raise ValueError('The result file must be a pkl file.')

    if args.out_dir is not None:
        os.makedirs(args.out_dir, exist_ok=True)

    print_log('This script visualizes the error for multiple object tracking. '
              'By Default, the red bounding box denotes false positive, '
              'the yellow bounding box denotes the false negative '
              'and the blue bounding box denotes ID switch.')

    cfg = Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.val, dict(test_mode=True))
    results = mmcv.load(args.result_file)

    # create index from frame_id to filename
    filenames_dict = dict()
    for data_info in dataset.data_infos:
        video_name = data_info['filename'].split(os.sep, 1)[0]
        frame_id = int(data_info['filename'].rsplit(os.sep,
                                                    1)[-1].split('.')[0])
        if video_name not in filenames_dict:
            filenames_dict[video_name] = dict()
        filenames_dict[video_name][frame_id] = data_info['filename']

    # format the results to txts
    resfile_path, resfiles, video_names, tmp_dir = dataset.format_results(
        results, None, ['track'])

    for video_name in video_names:
        print_log(f'Start processing video {video_name}')

        acc, res, gt = compare_res_gts(resfiles, dataset, video_name)

        frames_id_list = sorted(
            list(set(acc.mot_events.index.get_level_values(0))))
        for frame_id in frames_id_list:
            # events in the current frame
            events = acc.mot_events.xs(frame_id)
            cur_res = res.loc[frame_id] if frame_id in res.index else None
            cur_gt = gt.loc[frame_id] if frame_id in gt.index else None
            # path of image
            img = osp.join(dataset.img_prefix,
                           filenames_dict[video_name][frame_id])
            fps = events[events.Type == 'FP']
            fns = events[events.Type == 'MISS']
            idsws = events[events.Type == 'SWITCH']

            bboxes, ids, error_types = [], [], []
            for fp_index in fps.index:
                hid = events.loc[fp_index].HId
                bboxes.append([
                    cur_res.loc[hid].X, cur_res.loc[hid].Y,
                    cur_res.loc[hid].X + cur_res.loc[hid].Width,
                    cur_res.loc[hid].Y + cur_res.loc[hid].Height,
                    cur_res.loc[hid].Confidence
                ])
                ids.append(hid)
                # error_type = 0 denotes false positive error
                error_types.append(0)
            for fn_index in fns.index:
                oid = events.loc[fn_index].OId
                bboxes.append([
                    cur_gt.loc[oid].X, cur_gt.loc[oid].Y,
                    cur_gt.loc[oid].X + cur_gt.loc[oid].Width,
                    cur_gt.loc[oid].Y + cur_gt.loc[oid].Height,
                    cur_gt.loc[oid].Confidence
                ])
                ids.append(-1)
                # error_type = 1 denotes false negative error
                error_types.append(1)
            for idsw_index in idsws.index:
                hid = events.loc[idsw_index].HId
                bboxes.append([
                    cur_res.loc[hid].X, cur_res.loc[hid].Y,
                    cur_res.loc[hid].X + cur_res.loc[hid].Width,
                    cur_res.loc[hid].Y + cur_res.loc[hid].Height,
                    cur_res.loc[hid].Confidence
                ])
                ids.append(hid)
                # error_type = 2 denotes id switch
                error_types.append(2)
            if len(bboxes) == 0:
                bboxes = np.zeros((0, 5), dtype=np.float32)
            else:
                bboxes = np.asarray(bboxes, dtype=np.float32)
            ids = np.asarray(ids, dtype=np.int32)
            error_types = np.asarray(error_types, dtype=np.int32)
            imshow_mot_errors(
                img,
                bboxes,
                ids,
                error_types,
                show=args.show,
                out_file=osp.join(args.out_dir,
                                  f'{video_name}/{frame_id:06d}.jpg')
                if args.out_dir else None,
                backend=args.backend)

        print_log(f'Done! Visualization images are saved in '
                  f'\'{args.out_dir}/{video_name}\'')

        mmcv.frames2video(
            f'{args.out_dir}/{video_name}',
            f'{args.out_dir}/{video_name}.mp4',
            fps=args.fps,
            fourcc='mp4v',
            start=frames_id_list[0],
            end=frames_id_list[-1],
            show_progress=False)
        print_log(
            f'Done! Visualization video is saved as '
            f'\'{args.out_dir}/{video_name}.mp4\' with a FPS of {args.fps}')


if __name__ == '__main__':
    main()
