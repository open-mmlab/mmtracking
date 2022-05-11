# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmtrack.models.motion import FlowNetSimple


def test_flownet_simple():
    # Test flownet_simple forward
    model = FlowNetSimple(img_scale_factor=0.5)
    model.init_weights()
    model.train()

    imgs = torch.randn(2, 6, 224, 224)
    img_metas = [
        dict(
            img_norm_cfg=dict(
                mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375)),
            img_shape=(224, 224, 3))
    ]
    flow = model(imgs, img_metas)
    assert flow.shape == torch.Size([2, 2, 224, 224])
