# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import tempfile
from argparse import ArgumentParser

import mmcv
import mmengine

from mmtrack.apis import inference_mot, init_model
from mmpose.structures import merge_data_samples
from mmtrack.registry import VISUALIZERS
from mmtrack.utils import register_all_modules
from mmpose.utils import register_all_modules as register_all_modules_pose


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('config', help='config file')
    parser.add_argument('--input', help='input video file or folder')
    parser.add_argument(
        '--output', help='output video file (mp4 format) or folder')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.0,
        help='The threshold of score to filter bboxes.')
    parser.add_argument(
        '--device', default='cuda:0', help='device used for inference')
    parser.add_argument(
        '--show',
        action='store_true',
        help='whether show the results on the fly')
    parser.add_argument('--fps', help='FPS of the output video')
    args = parser.parse_args()
    return args


def draw_image(pose, img):
    print('number of keypoints:', len(pose))
    import cv2

    for k in range(len(pose)):
        landmarks = pose[k].pred_instances.keypoints.reshape(-1, 2)

        for i in range(landmarks.shape[0]):
            center_coordinates = (int(landmarks[i][0]), int(landmarks[i][1]))
            radius = 3
            color = (100, 255, 100)
            thickness = 1
            img = cv2.circle(img, center_coordinates, radius, color, thickness)

        cv2.imwrite('image2.jpg', img)


def main(args):
    assert args.output or args.show
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
    if args.show or OUT_VIDEO:
        if fps is None and IN_VIDEO:
            fps = imgs.fps
        if not fps:
            raise ValueError('Please set the FPS for the output video.')
        fps = int(fps)

    register_all_modules(init_default_scope=True)
    register_all_modules_pose(init_default_scope=False)

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)

    # build the visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = get_dataset_meta()

    # build the pose visualizer
    from mmpose.visualization import PoseLocalVisualizer
    pose_visualizer = VISUALIZERS.build(
        dict(
            type='PoseLocalVisualizer',
            name='visualizer',
            radius=3,
            line_width=1))

    prog_bar = mmengine.ProgressBar(len(imgs))
    # test and show/save the images
    for i, img in enumerate(imgs):
        if isinstance(img, str):
            img_path = osp.join(args.input, img)
            img = mmcv.imread(img_path)

        print()
        print('origin image:', img.shape)
        result = inference_mot(model, img, frame_id=i)

        if args.output is not None:
            if IN_VIDEO or OUT_VIDEO:
                out_file = osp.join(out_path, f'{i:06d}.jpg')
            else:
                out_file = osp.join(out_path, img.rsplit(os.sep, 1)[-1])
        else:
            out_file = None

        pose_result = result.pred_track_instances.pose
        # draw keypoints
        draw_image(pose_result, img.copy())
        data_samples = merge_data_samples(pose_result)

        # show the results
        visualizer.add_datasample(
            'mot',
            img[..., ::-1],
            data_sample=result,
            show=args.show,
            draw_gt=False,
            out_file=out_file,
            wait_time=float(1 / int(fps)) if fps else 0,
            pred_score_thr=args.score_thr,
            step=i)

        prog_bar.update()

    if args.output and OUT_VIDEO:
        print(f'making the output video at {args.output} with a FPS of {fps}')
        mmcv.frames2video(out_path, args.output, fps=fps, fourcc='mp4v')
        out_dir.cleanup()


def get_dataset_meta():
    dataset_info = dict(
        dataset_name='coco',
        paper_info=dict(
            author='Lin, Tsung-Yi and Maire, Michael and '
            'Belongie, Serge and Hays, James and '
            'Perona, Pietro and Ramanan, Deva and '
            r'Doll{\'a}r, Piotr and Zitnick, C Lawrence',
            title='Microsoft coco: Common objects in context',
            container='European conference on computer vision',
            year='2014',
            homepage='http://cocodataset.org/',
        ),
        keypoint_info={
            0:
            dict(
                name='nose', id=0, color=[51, 153, 255], type='upper',
                swap=''),
            1:
            dict(
                name='left_eye',
                id=1,
                color=[51, 153, 255],
                type='upper',
                swap='right_eye'),
            2:
            dict(
                name='right_eye',
                id=2,
                color=[51, 153, 255],
                type='upper',
                swap='left_eye'),
            3:
            dict(
                name='left_ear',
                id=3,
                color=[51, 153, 255],
                type='upper',
                swap='right_ear'),
            4:
            dict(
                name='right_ear',
                id=4,
                color=[51, 153, 255],
                type='upper',
                swap='left_ear'),
            5:
            dict(
                name='left_shoulder',
                id=5,
                color=[0, 255, 0],
                type='upper',
                swap='right_shoulder'),
            6:
            dict(
                name='right_shoulder',
                id=6,
                color=[255, 128, 0],
                type='upper',
                swap='left_shoulder'),
            7:
            dict(
                name='left_elbow',
                id=7,
                color=[0, 255, 0],
                type='upper',
                swap='right_elbow'),
            8:
            dict(
                name='right_elbow',
                id=8,
                color=[255, 128, 0],
                type='upper',
                swap='left_elbow'),
            9:
            dict(
                name='left_wrist',
                id=9,
                color=[0, 255, 0],
                type='upper',
                swap='right_wrist'),
            10:
            dict(
                name='right_wrist',
                id=10,
                color=[255, 128, 0],
                type='upper',
                swap='left_wrist'),
            11:
            dict(
                name='left_hip',
                id=11,
                color=[0, 255, 0],
                type='lower',
                swap='right_hip'),
            12:
            dict(
                name='right_hip',
                id=12,
                color=[255, 128, 0],
                type='lower',
                swap='left_hip'),
            13:
            dict(
                name='left_knee',
                id=13,
                color=[0, 255, 0],
                type='lower',
                swap='right_knee'),
            14:
            dict(
                name='right_knee',
                id=14,
                color=[255, 128, 0],
                type='lower',
                swap='left_knee'),
            15:
            dict(
                name='left_ankle',
                id=15,
                color=[0, 255, 0],
                type='lower',
                swap='right_ankle'),
            16:
            dict(
                name='right_ankle',
                id=16,
                color=[255, 128, 0],
                type='lower',
                swap='left_ankle')
        },
        skeleton_info={
            0:
            dict(link=('left_ankle', 'left_knee'), id=0, color=[0, 255, 0]),
            1:
            dict(link=('left_knee', 'left_hip'), id=1, color=[0, 255, 0]),
            2:
            dict(
                link=('right_ankle', 'right_knee'), id=2, color=[255, 128, 0]),
            3:
            dict(link=('right_knee', 'right_hip'), id=3, color=[255, 128, 0]),
            4:
            dict(link=('left_hip', 'right_hip'), id=4, color=[51, 153, 255]),
            5:
            dict(
                link=('left_shoulder', 'left_hip'), id=5, color=[51, 153,
                                                                 255]),
            6:
            dict(
                link=('right_shoulder', 'right_hip'),
                id=6,
                color=[51, 153, 255]),
            7:
            dict(
                link=('left_shoulder', 'right_shoulder'),
                id=7,
                color=[51, 153, 255]),
            8:
            dict(
                link=('left_shoulder', 'left_elbow'), id=8, color=[0, 255, 0]),
            9:
            dict(
                link=('right_shoulder', 'right_elbow'),
                id=9,
                color=[255, 128, 0]),
            10:
            dict(link=('left_elbow', 'left_wrist'), id=10, color=[0, 255, 0]),
            11:
            dict(
                link=('right_elbow', 'right_wrist'),
                id=11,
                color=[255, 128, 0]),
            12:
            dict(link=('left_eye', 'right_eye'), id=12, color=[51, 153, 255]),
            13:
            dict(link=('nose', 'left_eye'), id=13, color=[51, 153, 255]),
            14:
            dict(link=('nose', 'right_eye'), id=14, color=[51, 153, 255]),
            15:
            dict(link=('left_eye', 'left_ear'), id=15, color=[51, 153, 255]),
            16:
            dict(link=('right_eye', 'right_ear'), id=16, color=[51, 153, 255]),
            17:
            dict(
                link=('left_ear', 'left_shoulder'),
                id=17,
                color=[51, 153, 255]),
            18:
            dict(
                link=('right_ear', 'right_shoulder'),
                id=18,
                color=[51, 153, 255])
        },
        joint_weights=[
            1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2, 1.2,
            1.5, 1.5
        ],
        sigmas=[
            0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
            0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089
        ])
    return dataset_info


if __name__ == '__main__':
    args = parse_args()
    main(args)
