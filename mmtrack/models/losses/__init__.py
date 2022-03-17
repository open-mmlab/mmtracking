# Copyright (c) OpenMMLab. All rights reserved.
from .kl_loss import KLGridLoss, KLMCLoss
from .l2_loss import L2Loss
from .triplet_loss import TripletLoss

__all__ = ['L2Loss', 'TripletLoss', 'KLMCLoss', 'KLGridLoss']
