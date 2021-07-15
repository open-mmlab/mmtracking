import argparse
import os
import os.path as osp
import tempfile

import cv2
import mmcv
import motmetrics as mm
import numpy as np
from mmcv import Config
from mmdet.datasets import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='visualize the situation of false positive, '
        'false negative and ID switch for multiple object tracking')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--result-file', help='path of inference result')
    parser.add_argument(
        '--output', help='directory where painted images will be saved')
    parser.add_argument(
        '--show', action='store_true', help='show visualization results')
    parser.add_argument('--fps', help='FPS of the output video')
    args = parser.parse_args()
    return args


def show_wrong_tracks(img,
                      bboxes,
                      ids,
                      wrong_types,
                      bbox_colors=[(0, 0, 255), (0, 255, 255), (255, 0, 0)],
                      thickness=2,
                      font_scale=0.4,
                      show=False,
                      wait_time=0,
                      out_file=None):
    """Show the wrong tracks with opencv."""
    assert bboxes.ndim == 2
    assert ids.ndim == 1
    assert wrong_types.ndim == 1
    assert bboxes.shape[1] == 5
    assert len(bbox_colors) == 3
    if isinstance(img, str):
        img = mmcv.imread(img)

    img_shape = img.shape
    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])

    text_width, text_height = 10, 15
    for bbox, wrong_type, id in zip(bboxes, wrong_types, ids):
        x1, y1, x2, y2 = bbox[:4].astype(np.int32)
        score = float(bbox[-1])

        # bbox
        bbox_color = bbox_colors[wrong_type]
        cv2.rectangle(img, (x1, y1), (x2, y2), bbox_color, thickness=thickness)

        # FN and IDSW do not need id and score
        if wrong_type == 1 or wrong_type == 2:
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

    assert args.output or args.show, \
        ('Please specify at least one operation (show the results '
         '/ save the results) with the argument "--out-dir" or "--show"')

    if not args.result_file.endswith(('.pkl', 'pickle')):
        raise ValueError('The result file must be a txt file.')

    # define output
    if args.output is not None:
        if args.output.endswith('.mp4'):
            OUT_VIDEO = True
            out_dir = tempfile.TemporaryDirectory()
            out_path = out_dir.name
            _out = args.output.rsplit('/', 1)
            if len(_out) > 1:
                os.makedirs(_out[0], exist_ok=True)
        else:
            OUT_VIDEO = False
            out_path = args.output
            os.makedirs(out_path, exist_ok=True)

    cfg = Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.val, dict(test_mode=True))
    results = mmcv.load(args.result_file)

    resfiles, names, tmp_dir = dataset.format_results(results, None, ['track'])

    for name in ['MOT17-05-DPM']:
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
        first_frame_id = acc.mot_events.index[0][0]
        last_frame_id = acc.mot_events.index[-1][0]
        for frame_id in range(first_frame_id, last_frame_id + 1):
            # events in the current frame
            events = acc.mot_events.xs(frame_id)
            cur_res = res.loc[frame_id]
            cur_gt = gt.loc[frame_id]
            # path of img
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
                wrong_types.append(1)
            for idsw_index in idsws.index:
                hid = events.loc[idsw_index].HId
                bboxes.append([
                    cur_res.loc[hid].X, cur_res.loc[hid].Y,
                    cur_res.loc[hid].X + cur_res.loc[hid].Width,
                    cur_res.loc[hid].Y + cur_res.loc[hid].Height,
                    cur_res.loc[hid].Confidence
                ])
                ids.append(-1)
                wrong_types.append(2)
            if len(bboxes) == 0:
                bboxes = np.zeros(())
            bboxes = np.asarray(bboxes, dtype=np.float32)
            ids = np.asarray(ids, dtype=np.int32)
            wrong_types = np.asarray(wrong_types, dtype=np.int32)
            show_wrong_tracks(
                img,
                bboxes,
                ids,
                wrong_types,
                show=args.show,
                out_file=osp.join(out_path, f'{name}/{frame_id:06d}.jpg'))

        if OUT_VIDEO:
            print(
                f'making the output video at {args.output} with a FPS of {fps}'
            )
            mmcv.frames2video(out_path, args.output, fps=fps)
            out_dir.cleanup()


if __name__ == '__main__':
    main()
