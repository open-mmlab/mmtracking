from mmtrack.apis import inference_mot, init_model
from mmtrack.utils import register_all_modules
from mmcv import VideoReader

register_all_modules()
model = init_model("configs/mot/deepsort/my_config.py", device="cpu")
imgs = VideoReader("demo/demo.mp4")

inference_mot(model, imgs[0], 0)