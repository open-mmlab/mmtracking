# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import pytest
import torch

from mmtrack.models.layers import SinePositionalEncoding3D


class TestSinePositionalEncoding3D(TestCase):

    def test_sine_positional_encoding_3(self, num_feats=16, batch_size=2):
        # test invalid type of scale
        with pytest.raises(AssertionError):
            module = SinePositionalEncoding3D(
                num_feats, scale=(3., ), normalize=True)

        module = SinePositionalEncoding3D(num_feats)
        t, h, w = 2, 10, 6
        mask = (torch.rand(batch_size, t, h, w) > 0.5).to(torch.int)
        assert not module.normalize
        out = module(mask)
        assert out.shape == (batch_size, t, num_feats * 2, h, w)

        # set normalize
        module = SinePositionalEncoding3D(num_feats, normalize=True)
        assert module.normalize
        out = module(mask)
        assert out.shape == (batch_size, t, num_feats * 2, h, w)
