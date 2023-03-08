from mmengine.model import BaseModel

from mmtrack.registry import MODELS

from torchreid.reid.utils import FeatureExtractor


@MODELS.register_module()
class MyReID(BaseModel):

    def __init__(self, model_name: str, model_path: str, device: str):
        super().__init__()
        self.reid: FeatureExtractor = FeatureExtractor(
            model_name=model_name, model_path=model_path, device=device)

    @property
    def head(self):

        class Head:
            out_channels = self.reid.model.feature_dim

        return Head()

    def forward(self, inputs, mode: str = 'tensor'):
        return self.reid(inputs)