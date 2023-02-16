from mmtrack.apis import inference_mot, init_model
from mmtrack.registry import VISUALIZERS
from mmtrack.utils import register_all_modules

from mmcv import VideoReader

if __name__ == "__main__":
    register_all_modules()
    model = init_model(
        "configs/mot/qdtrack/qdtrack_yolov7.py", 
        # "configs/mot/qdtrack/qdtrack_faster-rcnn_r50_fpn_8xb2-4e_mot17halftrain_test-mot17halfval.py",
        device="cpu")
    imgs = VideoReader("demo/demo.mp4")

    inference_mot(model, imgs[0], 0)
