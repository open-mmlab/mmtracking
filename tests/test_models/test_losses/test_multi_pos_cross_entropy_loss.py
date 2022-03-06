# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmtrack.models import MultiPosCrossEntropyLoss


def test_mpce_loss():
    costs = torch.tensor([[1, 0], [0, 1]])
    labels = torch.tensor([[1, 1], [0, 0]])

    loss = MultiPosCrossEntropyLoss(reduction='mean', loss_weight=1.0)
    assert torch.allclose(loss(costs, labels), torch.tensor(0.))

    labels = torch.Tensor([[1, 0], [0, 1]])
    loss(costs, labels)
    assert torch.allclose(loss(costs, labels), torch.tensor(0.31326))
