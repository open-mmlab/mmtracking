import os, shutil
from warnings import warn
from tqdm import tqdm
from argparse import ArgumentParser

import mmcv


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("data_dir", help="Path to the data directory")
    parser.add_argument("--fps", type=int, default=5, help="FPS to extract images at. Default: 5")

    return parser.parse_args()

def extract_images(vid_path: str, imgs_dir: str, fps: int):
    """
    Extract images from a video and save them to a directory.

    Args:
        vid_path (str): Path to the video file.
        imgs_dir (str): Path to the directory where the images will be saved.
    """

    if not os.path.exists(vid_path):
        print(f"Missing video: {vid_path}")

    if os.path.exists(imgs_dir):
        warn(f"Images already extracted: {imgs_dir}. Removing...")
        shutil.rmtree(imgs_dir)
    os.makedirs(imgs_dir)

    reader = mmcv.VideoReader(vid_path)
    
    frame_step = int(reader.fps / fps)

    for frame_id in range(0, reader.frame_cnt, frame_step):
        frame = reader.get_frame(frame_id)
        mmcv.imwrite(frame, os.path.join(imgs_dir, f"{frame_id:06d}.jpg"))

def main():
    args = parse_args()

    for subset in ("train", "validation", "test"):
        subset_dir = os.path.join(args.data_dir, subset)

        print(f'Extracing images from {subset} set')
        for scene_dir in tqdm(os.scandir(subset_dir)):
            if scene_dir.is_dir():
                for camera_dir in os.scandir(scene_dir):
                    if camera_dir.is_dir():
                        vid_path = os.path.join(camera_dir.path, "video.mp4")
                        imgs_dir = os.path.join(camera_dir.path, "imgs")

                        extract_images(vid_path, imgs_dir, args.fps)


if __name__ == "__main__":
    main()
