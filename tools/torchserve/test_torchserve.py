# Copyright (c) OpenMMLab. All rights reserved.
import tempfile
from argparse import ArgumentParser

import cv2
import requests

from mmtrack.apis import inference_vid, init_model


def parse_args():
    parser = ArgumentParser(
        description='Compare result of torchserve and pytorch,'
        'and visualize them.')
    parser.add_argument('video', help='Video file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('model_name', help='The model name in the server')
    parser.add_argument(
        '--inference-addr',
        default='127.0.0.1:8080',
        help='Address and port of the inference server')
    parser.add_argument(
        '--result-video',
        type=str,
        default=None,
        help='save serve output in result-video')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.5, help='bbox score threshold')

    args = parser.parse_args()
    return args


def main(args):
    url = 'http://' + args.inference_addr + '/predictions/' + args.model_name
    with open(args.video, 'rb') as video:
        tmp_res = requests.post(url, video)
    content = tmp_res.content
    if args.result_video:
        with open(args.result_video, 'wb') as file:
            file.write(content)
        video_path = args.result_video
    else:
        video_path = tempfile.NamedTemporaryFile().name
        with open(video_path, 'wb') as file:
            file.write(content)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    while cap.isOpened():
        flag, frame = cap.read()
        if not flag:
            break
        cv2.imshow('torchserve_result', frame)
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    model = init_model(args.config, args.checkpoint, device=args.device)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_id = 0
    while cap.isOpened():
        flag, frame = cap.read()
        if not flag:
            break

        # test a single image
        result = inference_vid(model, frame, frame_id)
        vis_frame = model.show_result(
            frame, result, score_thr=args.score_thr, show=False)

        cv2.imshow('pytorch_result', vis_frame)

        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    args = parse_args()
    main(args)
