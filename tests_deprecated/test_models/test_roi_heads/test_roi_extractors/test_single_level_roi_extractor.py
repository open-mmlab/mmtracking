# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmtrack.models.roi_heads.roi_extractors import SingleRoIExtractor


def test_single_roi_extractor():
    """Tests single roi extractor."""
    single_roi_extractor_config = dict(
        roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32])
    self = SingleRoIExtractor(**single_roi_extractor_config)

    feats = (
        torch.rand((1, 256, 200, 336)),
        torch.rand((1, 256, 100, 168)),
        torch.rand((1, 256, 50, 84)),
        torch.rand((1, 256, 25, 42)),
    )

    rois = torch.tensor([[0.0000, 587.8285, 52.1405, 886.2484, 341.5644]])
    # test allowing to accept external arguments by **kwargs
    roi_feats = self(feats, rois, variable=1)
    assert roi_feats.shape == torch.Size([1, 256, 7, 7])
