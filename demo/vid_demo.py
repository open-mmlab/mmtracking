import os
from argparse import ArgumentParser

import cv2

from mmtrack.apis import init_vid_model, vid_inference


def main():
    parser = ArgumentParser()
    parser.add_argument('video', help='Video file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--out-video-root',
        default='',
        help='Root of the output video file. '
        'Default not saving the visualization video.')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show visualizations.')
    parser.add_argument(
        '--score-thr', type=float, default=0.8, help='bbox score threshold')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_vid_model(args.config, args.checkpoint, device=args.device)

    cap = cv2.VideoCapture(args.video)

    if args.out_video_root == '':
        save_out_video = False
    else:
        os.makedirs(args.out_video_root, exist_ok=True)
        save_out_video = True

        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video_path = os.path.join(args.out_video_root,
                                      f'vis_{os.path.basename(args.video)}')
        videoWriter = cv2.VideoWriter(out_video_path, fourcc, fps, size)

    frame_id = 0
    while (cap.isOpened()):
        flag, frame = cap.read()
        if not flag:
            break

        # test a single image
        result = vid_inference(model, frame, frame_id)
        vis_frame = model.show_result(
            frame, result, score_thr=args.score_thr, show=False)

        if save_out_video:
            videoWriter.write(vis_frame)

        if args.show:
            cv2.imshow(args.video, vis_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_id += 1

    cap.release()
    if save_out_video:
        videoWriter.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
