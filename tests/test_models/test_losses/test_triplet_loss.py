# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmtrack.models import TripletLoss


def test_triplet_loss():
    feature = torch.Tensor([[1, 1], [1, 1], [0, 0], [0, 0]])
    label = torch.Tensor([1, 1, 0, 0])

    loss = TripletLoss(margin=0.3, loss_weight=1.0)
    assert torch.allclose(loss(feature, label), torch.tensor(0.))

    label = torch.Tensor([1, 0, 1, 0])
    assert torch.allclose(loss(feature, label), torch.tensor(1.7142))
