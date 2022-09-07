# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import tempfile
from argparse import ArgumentParser

import cv2
import mmcv
import mmengine

from mmtrack.apis import inference_sot, init_model
from mmtrack.registry import VISUALIZERS
from mmtrack.utils import register_all_modules


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('--input', help='input video file')
    parser.add_argument('--output', help='output video file (mp4 format)')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show visualizations.')
    parser.add_argument('--fps', help='FPS of the output video')
    parser.add_argument('--gt_bbox_file', help='The path of gt_bbox file')
    args = parser.parse_args()
    return args


def main(args):
    # load images
    if osp.isdir(args.input):
        imgs = sorted(
            filter(lambda x: x.endswith(('.jpg', '.png', '.jpeg')),
                   os.listdir(args.input)),
            key=lambda x: int(x.split('.')[0]))
        IN_VIDEO = False
    else:
        imgs = mmcv.VideoReader(args.input)
        IN_VIDEO = True

    # define output
    OUT_VIDEO = False
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
    if OUT_VIDEO:
        if fps is None and IN_VIDEO:
            fps = imgs.fps
        if not fps:
            raise ValueError('Please set the FPS for the output video.')
        fps = int(fps)

    register_all_modules(init_default_scope=True)

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)

    # build the visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    prog_bar = mmengine.ProgressBar(len(imgs))
    # test and show/save the images
    for i, img in enumerate(imgs):
        if isinstance(img, str):
            img_path = osp.join(args.input, img)
            img = mmcv.imread(img_path)
        if i == 0:
            if args.gt_bbox_file is not None:
                bboxes = mmengine.list_from_file(args.gt_bbox_file)
                init_bbox = list(map(float, bboxes[0].split(',')))
            else:
                init_bbox = list(cv2.selectROI(args.input, img, False, False))
                cv2.destroyAllWindows()

            # convert (x1, y1, w, h) to (x1, y1, x2, y2)
            init_bbox[2] += init_bbox[0]
            init_bbox[3] += init_bbox[1]

        result = inference_sot(model, img, init_bbox, frame_id=i)
        if args.output is not None:
            if IN_VIDEO or OUT_VIDEO:
                out_file = osp.join(out_path, f'{i:06d}.jpg')
            else:
                out_file = osp.join(out_path, img_path.rsplit(os.sep, 1)[-1])
        else:
            out_file = None

        # show the results
        visualizer.add_datasample(
            'sot',
            img[..., ::-1],
            data_sample=result,
            show=args.show,
            draw_gt=False,
            out_file=out_file,
            wait_time=float(1 / int(fps)) if fps else 0,
            pred_score_thr=-100,
            step=i)

        prog_bar.update()

    if args.output and OUT_VIDEO:
        print(
            f'\nmaking the output video at {args.output} with a FPS of {fps}')
        mmcv.frames2video(out_path, args.output, fps=fps, fourcc='mp4v')
        out_dir.cleanup()


if __name__ == '__main__':
    args = parse_args()
    main(args)
