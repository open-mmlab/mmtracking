# This script visualizes the error for multiple object tracking.
#
# In painted images or videos, The yellow bounding box denotes false negative,
# the bounding box denotes the false positive and the green bounding box
# denotes ID switch.
import argparse
import os
import os.path as osp

import cv2
import mmcv
import motmetrics as mm
import numpy as np
import seaborn as sns
from mmcv import Config

from mmtrack.datasets import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='visualize the error for multiple object tracking')
    parser.add_argument('config', help='path of the config file')
    parser.add_argument('--result-file', help='path of inference result')
    parser.add_argument(
        '--out-dir',
        help='directory where painted images or videos will be saved')
    parser.add_argument(
        '--out-video', action='store_true', help='whether to output video')
    parser.add_argument(
        '--out-image', action='store_true', help='whether to output image')
    parser.add_argument(
        '--show',
        action='store_true',
        help='whether to show the results on the fly')
    parser.add_argument(
        '--fps', type=int, default=3, help='FPS of the output video')
    args = parser.parse_args()
    return args


def show_wrong_tracks(img,
                      bboxes,
                      ids,
                      wrong_types,
                      bbox_colors=None,
                      thickness=2,
                      font_scale=0.4,
                      text_width=10,
                      text_height=15,
                      show=False,
                      wait_time=0,
                      out_file=None):
    """Show the wrong tracks with opencv.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): A ndarray of shape (k, 5).
        ids (ndarray): A ndarray of shape (k, ).
        wrong_types (ndarray): A ndarray of shape (k, ), where 0 denotes
            false positives, 1 denotes false negative and 2 denotes ID switch.
        bbox_colors (list[tuple], optional): A list of colors to
            draw boxes with different wrong type. Defaults to None.
        thickness (int, optional): Thickness of lines.
            Defaults to 2.
        font_scale (float, optional): Font scale to draw id and score.
            Defaults to 0.4.
        text_width (int, optional): Width to draw id and score.
            Defaults to 10.
        text_height (int, optional): Height to draw id and score.
            Defaults to 15.
        show (bool, optional): Whether to show the image on the fly.
            Defaults to False.
        wait_time (int, optional): Value of waitKey param.
            Defaults to 0.
        out_file (str, optional): The filename to write the image.
            Defaults to None.
    """
    assert bboxes.ndim == 2
    assert ids.ndim == 1
    assert wrong_types.ndim == 1
    assert bboxes.shape[1] == 5

    if bbox_colors:
        assert len(bbox_colors) == 3
    else:
        bbox_colors = sns.color_palette(n_colors=3)
        bbox_colors = [[int(255 * _c) for _c in bbox_color][::-1]
                       for bbox_color in bbox_colors]
    if isinstance(img, str):
        img = mmcv.imread(img)

    img_shape = img.shape
    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])

    for bbox, wrong_type, id in zip(bboxes, wrong_types, ids):
        x1, y1, x2, y2 = bbox[:4].astype(np.int32)
        score = float(bbox[-1])

        # bbox
        bbox_color = bbox_colors[wrong_type]
        cv2.rectangle(img, (x1, y1), (x2, y2), bbox_color, thickness=thickness)

        # FN does not have id and score
        if wrong_type == 1:
            continue

        # id
        text = str(id)
        width = len(text) * text_width
        img[y1:y1 + text_height, x1:x1 + width, :] = bbox_color
        cv2.putText(
            img,
            str(id), (x1, y1 + text_height - 2),
            cv2.FONT_HERSHEY_COMPLEX,
            font_scale,
            color=(0, 0, 0))

        # score
        text = '{:.02f}'.format(score)
        width = len(text) * text_width
        img[y1 - text_height:y1, x1:x1 + width, :] = bbox_color
        cv2.putText(
            img,
            text, (x1, y1 - 2),
            cv2.FONT_HERSHEY_COMPLEX,
            font_scale,
            color=(0, 0, 0))

    if show:
        mmcv.imshow(img, wait_time=wait_time)
    if out_file is not None:
        mmcv.imwrite(img, out_file)

    return img


def main():
    args = parse_args()

    assert args.out_dir or args.show, \
        ('Please specify at least one operation (show the results '
         '/ save the results) with the argument "--show" or "--out-dir"')

    if args.out_dir:
        assert args.out_image or args.out_video, \
            ('Please specify at least one type (save as images save as videos)'
             ' with the argument "--out-image" or "--out-video"')

    if not args.result_file.endswith(('.pkl', 'pickle')):
        raise ValueError('The result file must be a pkl file.')

    if args.out_dir is not None:
        os.makedirs(args.out_dir, exist_ok=True)

    cfg = Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.val, dict(test_mode=True))
    results = mmcv.load(args.result_file)

    # format the results to txts
    resfiles, names, tmp_dir = dataset.format_results(results, None, ['track'])

    for name in names:
        print(f'Start processing video {name}')
        if 'half-train' in dataset.ann_file:
            gt_file = osp.join(dataset.img_prefix,
                               f'{name}/gt/gt_half-train.txt')
        elif 'half-val' in dataset.ann_file:
            gt_file = osp.join(dataset.img_prefix,
                               f'{name}/gt/gt_half-val.txt')
        else:
            gt_file = osp.join(dataset.img_prefix, f'{name}/gt/gt.txt')
        res_file = osp.join(resfiles['track'], f'{name}.txt')
        gt = mm.io.loadtxt(gt_file)
        res = mm.io.loadtxt(res_file)
        ini_file = osp.join(dataset.img_prefix, f'{name}/seqinfo.ini')
        if osp.exists(ini_file):
            acc, ana = mm.utils.CLEAR_MOT_M(gt, res, ini_file)
        else:
            acc = mm.utils.compare_to_groundtruth(gt, res)

        infos = mmcv.list_from_file(ini_file)
        width = int(infos[5].strip().split('=')[1])
        height = int(infos[6].strip().split('=')[1])
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        if args.out_video:
            video_writer = cv2.VideoWriter(
                osp.join(args.out_dir, f'{name}.mp4'), fourcc, args.fps,
                (width, height))
        frame_id_list = list(set(acc.mot_events.index.get_level_values(0)))
        for frame_id in frame_id_list:
            # events in the current frame
            events = acc.mot_events.xs(frame_id)
            cur_res = res.loc[frame_id] if frame_id in res.index else None
            cur_gt = gt.loc[frame_id] if frame_id in gt.index else None
            # path of image
            img = osp.join(dataset.img_prefix,
                           f'{name}/img1/{frame_id:06d}.jpg')
            fps = events[events.Type == 'FP']
            fns = events[events.Type == 'MISS']
            idsws = events[events.Type == 'SWITCH']

            bboxes, ids, wrong_types = [], [], []
            for fp_index in fps.index:
                hid = events.loc[fp_index].HId
                bboxes.append([
                    cur_res.loc[hid].X, cur_res.loc[hid].Y,
                    cur_res.loc[hid].X + cur_res.loc[hid].Width,
                    cur_res.loc[hid].Y + cur_res.loc[hid].Height,
                    cur_res.loc[hid].Confidence
                ])
                ids.append(hid)
                # wrong_type = 0 denotes false positive error
                wrong_types.append(0)
            for fn_index in fns.index:
                oid = events.loc[fn_index].OId
                bboxes.append([
                    cur_gt.loc[oid].X, cur_gt.loc[oid].Y,
                    cur_gt.loc[oid].X + cur_gt.loc[oid].Width,
                    cur_gt.loc[oid].Y + cur_gt.loc[oid].Height,
                    cur_gt.loc[oid].Confidence
                ])
                ids.append(-1)
                # wrong_type = 1 denotes false negative error
                wrong_types.append(1)
            for idsw_index in idsws.index:
                hid = events.loc[idsw_index].HId
                bboxes.append([
                    cur_res.loc[hid].X, cur_res.loc[hid].Y,
                    cur_res.loc[hid].X + cur_res.loc[hid].Width,
                    cur_res.loc[hid].Y + cur_res.loc[hid].Height,
                    cur_res.loc[hid].Confidence
                ])
                ids.append(hid)
                # wrong_type = 2 denotes id switch
                wrong_types.append(2)
            if len(bboxes) == 0:
                bboxes = np.zeros((0, 5), dtype=np.float32)
            else:
                bboxes = np.asarray(bboxes, dtype=np.float32)
            ids = np.asarray(ids, dtype=np.int32)
            wrong_types = np.asarray(wrong_types, dtype=np.int32)
            vis_frame = show_wrong_tracks(
                img,
                bboxes,
                ids,
                wrong_types,
                show=args.show,
                out_file=osp.join(args.out_dir, f'{name}/{frame_id:06d}.jpg')
                if args.out_image else None)
            if args.out_video:
                video_writer.write(vis_frame)

        if args.out_video:
            print(f'Done! Visualization video is saved as '
                  f'\'{args.out_dir}/{name}.mp4\' with a FPS of {args.fps}')
            video_writer.release()
        if args.out_image:
            print(f'Done! Visualization images are saved in '
                  f'\'{args.out_dir}/{name}\'')


if __name__ == '__main__':
    main()
