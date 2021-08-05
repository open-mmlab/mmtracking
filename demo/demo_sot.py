from argparse import ArgumentParser

import cv2

from mmtrack.apis import inference_sot, init_model


def main():
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
    parser.add_argument(
        '--color', default=(0, 255, 0), help='Color of tracked bbox lines.')
    parser.add_argument(
        '--thickness', default=3, type=int, help='Thickness of bbox lines.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)

    cap = cv2.VideoCapture(args.input)

    save_out_video = False
    if args.output is not None:
        save_out_video = True

        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(args.output, fourcc, fps, size)

    frame_id = 0
    while (cap.isOpened()):
        flag, frame = cap.read()
        if not flag:
            break

        if frame_id == 0:
            init_bbox = list(cv2.selectROI(args.input, frame, False, False))
            # convert (x1, y1, w, h) to (x1, y1, x2, y2)
            init_bbox[2] += init_bbox[0]
            init_bbox[3] += init_bbox[1]
        # test a single image
        result = inference_sot(model, frame, init_bbox, frame_id)

        vis_frame = model.show_result(
            frame,
            result,
            color=args.color,
            thickness=args.thickness,
            show=False)

        if save_out_video:
            videoWriter.write(vis_frame)

        if args.show:
            cv2.imshow(args.input, vis_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_id += 1

    cap.release()
    if save_out_video:
        videoWriter.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
