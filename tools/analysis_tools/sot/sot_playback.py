# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import tempfile
from argparse import ArgumentParser

import mmcv
import mmengine
import numpy as np


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('video_path', help='video path')
    parser.add_argument('track_results', help='the tracked results')
    parser.add_argument('--gt_bboxes', help='the groundtruth bboxes file')
    parser.add_argument('--output', help='output video file (mp4 format)')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show visualizations.')
    parser.add_argument('--fps', help='FPS of the output video')
    args = parser.parse_args()
    return args


def main(args):

    # load images
    if osp.isdir(args.video_path):
        imgs = sorted(
            filter(lambda x: x.endswith(('.jpg', '.png', '.jpeg')),
                   os.listdir(args.video_path)),
            key=lambda x: int(x.split('.')[0]))
        IN_VIDEO = False
    else:
        imgs = mmcv.VideoReader(args.video_path)
        IN_VIDEO = True

    OUT_VIDEO = False
    # define output
    if args.output is not None:
        if args.output.endswith('.mp4'):
            OUT_VIDEO = True
            out_dir = tempfile.TemporaryDirectory()
            out_path = out_dir.name
            _out = args.output.rsplit(os.sep, 1)
            if len(_out) > 1:
                os.makedirs(_out[0], exist_ok=True)
        else:
            out_path = args.output
            os.makedirs(out_path, exist_ok=True)
    fps = args.fps
    if args.show or OUT_VIDEO:
        if fps is None:
            if IN_VIDEO:
                fps = imgs.fps
            if OUT_VIDEO:
                raise ValueError('Please set the FPS for the output video.')
        else:
            fps = int(fps)

    prog_bar = mmengine.ProgressBar(len(imgs))
    track_bboxes = mmengine.list_from_file(args.track_results)
    if args.gt_bboxes is not None:
        gt_bboxes = mmengine.list_from_file(args.gt_bboxes)
        assert len(track_bboxes) == len(gt_bboxes)

    # test and show/save the images
    for i, img in enumerate(imgs):
        if isinstance(img, str):
            img_path = osp.join(args.video_path, img)
            img = mmcv.imread(img_path)

        if args.output is not None:
            if IN_VIDEO or OUT_VIDEO:
                out_file = osp.join(out_path, f'{i:06d}.jpg')
            else:
                out_file = osp.join(out_path, img_path.rsplit(os.sep, 1)[-1])
        else:
            out_file = None

        draw_bboxes = []
        track_bbox = np.array(list(map(float,
                                       track_bboxes[i].split(','))))[None]
        track_bbox[:, 2] += track_bbox[:, 0]
        track_bbox[:, 3] += track_bbox[:, 1]
        draw_bboxes.append(track_bbox)
        colors = 'green'
        if args.gt_bboxes is not None:
            gt_bbox = np.array(list(map(float, gt_bboxes[i].split(','))))[None]
            gt_bbox[:, 2] += gt_bbox[:, 0]
            gt_bbox[:, 3] += gt_bbox[:, 1]
            draw_bboxes.append(gt_bbox)
            colors = ['green', 'blue']

        mmcv.imshow_bboxes(
            img,
            draw_bboxes,
            show=args.show,
            colors=colors,
            wait_time=int(1000. / fps) if fps else 0,
            out_file=out_file,
            thickness=2)
        prog_bar.update()

    if args.output and OUT_VIDEO:
        print(
            f'\nmaking the output video at {args.output} with a FPS of {fps}')
        mmcv.frames2video(out_path, args.output, fps=fps, fourcc='mp4v')
        out_dir.cleanup()


if __name__ == '__main__':
    args = parse_args()
    main(args)
