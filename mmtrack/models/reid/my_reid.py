from mmengine.model import BaseModule

from mmtrack.registry import MODELS

from torchreid.reid.utils import FeatureExtractor


@MODELS.register_module()
class MyReID(BaseModule):

    def __init__(self, model_name: str, model_path: str, device: str):
        reid: FeatureExtractor = FeatureExtractor(
            model_name=model_name, model_path=model_path, device=device)

    @property
    def head(self):
        return 256

    def forward(self, inputs):
        return self.reid(inputs)