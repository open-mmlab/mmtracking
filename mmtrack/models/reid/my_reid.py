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
        # norm reid:
        # norm_mean=[0.485, 0.456, 0.406]
        # norm_std=[0.229, 0.224, 0.225]

        # norm deepsort: RGB
        # mean=[123.675, 116.28, 103.53]
        # std = [58.395, 57.12, 57.375]

        import torch
        mean = torch.tensor([123.675, 116.28, 103.53],
                            device=inputs.device).reshape(1, 3, 1, 1)
        std = torch.tensor([58.395, 57.12, 57.375],
                           device=inputs.device).reshape(1, 3, 1, 1)

        if (inputs.max() > 1): inputs = (inputs - mean) / std

        return self.reid(inputs)