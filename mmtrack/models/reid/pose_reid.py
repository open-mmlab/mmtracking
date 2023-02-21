from typing import Optional, List
from mmtrack.registry import MODELS
from mmtrack.structures import ReIDDataSample
from mmengine.model import BaseModel
import torch
from mmpose.structures import PoseDataSample
from mmengine.structures import InstanceData
from mmengine.dataset import Compose
import numpy as np
from mmpose.datasets.transforms import LoadImage, GetBBoxCenterScale, PackPoseInputs

@MODELS.register_module()
class PoseReID(BaseModel):
    def __init__(self, base_reid: Optional[dict] = None, pose_model: Optional[dict] = None, **kwargs):
        super().__init__(**kwargs)
    
        self.base_reid = MODELS.build(base_reid)
        self.pose_model = MODELS.build(pose_model)

        self.pose_pipeline = Compose([
            LoadImage(),
            GetBBoxCenterScale(),
            PackPoseInputs()
        ])
    
    def forward(self, 
                inputs: torch.Tensor,
                data_samples: Optional[List[ReIDDataSample]] = None,
                mode: str = 'tensor'):
        if mode == 'tensor':
            reid_results = self.base_reid(inputs, data_samples, mode)

            pose_data = []
            for input in inputs:
                img = input.detach().moveaxis(0, -1).cpu().numpy()
                height, width, _ = img.shape
                bboxes = np.array([[0, 0, width, height]], dtype=np.float32)
                pds = self.pose_pipeline(dict(
                    img=img,
                    bbox=bboxes))
                pds = pds['data_samples']

                pds.gt_instances.bbox_scores = np.ones((1))
                pds.set_field(input.shape[1:], 'input_size', field_type='metainfo')
                pds.set_field([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15], 'flip_indices', field_type='metainfo')
                
                pose_data.append(pds)

            pose_results = self.pose_model.predict(inputs, pose_data)
        else:
            raise NotImplementedError(f'PoseReID does not support mode {mode}')

        return reid_results
