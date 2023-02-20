import os
import cv2
from tqdm import tqdm
from argparse import ArgumentParser
import mmengine


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("data_dir", help="Path to the data directory")

    return parser.parse_args()

def parse_gts(ann_path: str):
    """
    Read the annotations from the ground-truth file and convert them to format.
        list of dicts with keys: `id`, `category_id`, `instance_id`, `bbox`, `area`
    Note: The `bbox` is in the format `[xtop, ytop, w, h]`.

    Args:
        ann_path (str): Path to the annotation file.
        Note: Each line in the annotation file is in the following format:
            `<frame_id>,<track_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,1,-1,-1,-1`

    Returns:
        list: List of annotations.
    """

    outs = []

    with open(ann_path, "r") as f:        
        while True:
            ann = f.readline()

            if ann == "":
                break

            ann = ann.strip().split(",")
            frame_id, instance_id = map(int, ann[:2])
            bbox = list(map(float, ann[2:6]))
            category_id = 1
            area = bbox[2] * bbox[3]
            
            ann = dict(
                id=frame_id,
                category_id=category_id,
                instance_id=instance_id,
                bbox=bbox,
                area=area)

            outs.append(ann)

    return outs

def get_image_infos(frames_dir: str):
    """
    Get the frames information from the directory containing the frame images.

    Args:
        frames_dir (str): Path to the directory containing the images.

    Returns:
        list: List of image information order by frame_id. Each element is a dict with keys `id`, `file_name`, `height`, `width`, `frame_id`.
    """

    outs = []

    height, width = None, None
    
    prev_frame_id = -1
    for img_path in os.scandir(frames_dir):
        frame_id = int(img_path.name.split(".")[0])

        assert frame_id > prev_frame_id, f"Frame ids are not in order: {frame_id} <= {prev_frame_id}"
        prev_frame_id = frame_id

        if height is None:
            height, width = cv2.imread(img_path.path).shape[:2]
        
        info = dict(
            file_name=img_path.path,
            height=height,
            width=width,
            id=frame_id)

        outs.append(info)
    
    return outs


def main():
    args = parse_args()

    for subset in ("train", "validation"):
        subset_anns = {
            "videos": [],
            "images": [],
            "annotations": [],
            "categories": []
        }
        subset_dir = os.path.join(args.data_dir, subset)
        save_dir = os.path.join(args.data_dir, "annotations")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        print(f'Extracting images from {subset} set')
        for scene_dir in tqdm(os.scandir(subset_dir)):
            if scene_dir.is_dir():
                for camera_id, camera_dir in enumerate(os.scandir(scene_dir)):
                    if camera_dir.is_dir():
                        imgs_dir = os.path.join(camera_dir.path, "imgs")
                        gt_path = os.path.join(camera_dir.path, "label.txt")

                        # Read the annotations
                        anns = parse_gts(gt_path)
                        imgs = get_image_infos(imgs_dir)

                        # Match the annotations with the image infos, and add `id` and `frame_id` keys to both of them
                        # Since frames are not extracted with true FPS, we need to match the annotations with the image infos
                        ann_id = 0
                        new_anns = []
                        for frame_id, imgs in enumerate(imgs):
                            img["frame_id"] = frame_id

                            while ann_id < len(anns):
                                ann = anns[ann_id]
                                if ann["id"] == img["id"]:
                                    ann["frame_id"] = frame_id
                                    new_anns.append(ann)
                                
                                ann_id += 1
                        anns = new_anns

                        # Add video_id keys to the annotations
                        for ann in anns:
                            ann["video_id"] = camera_id
                        
                        # Add video_id keys to the image infos
                        for img in imgs:
                            img["video_id"] = camera_id

                        # Add the annotations and image infos to the subset
                        subset_anns["annotations"].extend(anns)
                        subset_anns["images"].extend(imgs)

                        # Add the videos to the subset
                        subset_anns["videos"].append(
                            dict(
                                id=camera_id,
                                name=camera_dir.path,))
                        
        # Add the categories to the subset
        subset_anns["categories"].append(dict(id=1, name="person"))

        print("Saving annotations...")
        mmengine.dump(subset_anns, os.path.join(save_dir, f"{subset}.json"))
