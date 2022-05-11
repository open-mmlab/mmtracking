# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmtrack.models.roi_heads.roi_extractors import TemporalRoIAlign


def test_temporal_roi_align():
    """Test Temporal RoI Align."""
    temporal_roi_align_config = dict(
        num_most_similar_points=2,
        num_temporal_attention_blocks=4,
        roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),
        out_channels=256,
        featmap_strides=[16])
    self = TemporalRoIAlign(**temporal_roi_align_config)

    feats = (torch.rand((1, 256, 50, 84)), )
    ref_feats = (feats[0].repeat((2, 1, 1, 1)), )
    rois = torch.tensor([[0.0000, 587.8285, 52.1405, 886.2484, 341.5644]])

    # test when ref_feats is not None
    roi_feats = self(feats, rois, ref_feats=ref_feats)
    assert roi_feats.shape == torch.Size([1, 256, 7, 7])

    # test when ref_feats is None
    roi_feats = self(feats, rois, ref_feats=None)
    assert roi_feats.shape == torch.Size([1, 256, 7, 7])
