from mmtrack.apis import inference_mot, init_model
from mmtrack.registry import VISUALIZERS
from mmtrack.utils import register_all_modules

from mmcv import VideoReader

if __name__ == '__main__':
    register_all_modules()
    model = init_model("configs/mot/ocsort/ocsort_faster-rcnn_r50_fpn.py", device="cpu")
    imgs = VideoReader("demo/demo.mp4")

    inference_mot(model, imgs[0], 0)