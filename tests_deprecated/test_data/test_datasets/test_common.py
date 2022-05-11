# Copyright (c) OpenMMLab. All rights reserved.
import logging
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from mmcv.runner import EpochBasedRunner
from torch.utils.data import DataLoader

from mmtrack.core.evaluation import DistEvalHook, EvalHook


@patch('mmtrack.apis.single_gpu_test', MagicMock)
@patch('mmtrack.apis.multi_gpu_test', MagicMock)
@pytest.mark.parametrize('EvalHookParam', (EvalHook, DistEvalHook))
def test_evaluation_hook(EvalHookParam):
    # create dummy data
    dataloader = DataLoader(torch.ones((5, 2)))
    dataloader.dataset.load_as_video = True

    # 0.1. dataloader is not a DataLoader object
    with pytest.raises(TypeError):
        EvalHookParam(dataloader=MagicMock(), interval=-1)

    # 0.2. negative interval
    with pytest.raises(ValueError):
        EvalHookParam(dataloader, interval=-1)

    # 1. start=None, interval=1: perform evaluation after each epoch.
    runner = _build_demo_runner()
    evalhook = EvalHookParam(dataloader, interval=1)
    evalhook.evaluate = MagicMock()
    runner.register_hook(evalhook)
    runner.run([dataloader], [('train', 1)], 2)
    assert evalhook.evaluate.call_count == 2  # after epoch 1 & 2

    # 2. start=1, interval=1: perform evaluation after each epoch.
    runner = _build_demo_runner()

    evalhook = EvalHookParam(dataloader, start=1, interval=1)
    evalhook.evaluate = MagicMock()
    runner.register_hook(evalhook)
    runner.run([dataloader], [('train', 1)], 2)
    assert evalhook.evaluate.call_count == 2  # after epoch 1 & 2

    # 3. start=None, interval=2: perform evaluation after epoch 2, 4, 6, etc
    runner = _build_demo_runner()
    evalhook = EvalHookParam(dataloader, interval=2)
    evalhook.evaluate = MagicMock()
    runner.register_hook(evalhook)
    runner.run([dataloader], [('train', 1)], 2)
    assert evalhook.evaluate.call_count == 1  # after epoch 2

    # 4. start=1, interval=2: perform evaluation after epoch 1, 3, 5, etc
    runner = _build_demo_runner()
    evalhook = EvalHookParam(dataloader, start=1, interval=2)
    evalhook.evaluate = MagicMock()
    runner.register_hook(evalhook)
    runner.run([dataloader], [('train', 1)], 3)
    assert evalhook.evaluate.call_count == 2  # after epoch 1 & 3

    # 5. start=0/negative, interval=1: perform evaluation after each epoch and
    #    before epoch 1.
    runner = _build_demo_runner()
    evalhook = EvalHookParam(dataloader, start=0)
    evalhook.evaluate = MagicMock()
    runner.register_hook(evalhook)
    runner.run([dataloader], [('train', 1)], 2)
    assert evalhook.evaluate.call_count == 3  # before epoch1 and after e1 & e2

    # the evaluation start epoch cannot be less than 0
    runner = _build_demo_runner()
    with pytest.raises(ValueError):
        EvalHookParam(dataloader, start=-2)

    # 6. resuming from epoch i, start = x (x<=i), interval =1: perform
    #    evaluation after each epoch and before the first epoch.
    runner = _build_demo_runner()
    evalhook = EvalHookParam(dataloader, start=1)
    evalhook.evaluate = MagicMock()
    runner.register_hook(evalhook)
    runner._epoch = 2
    runner.run([dataloader], [('train', 1)], 3)
    assert evalhook.evaluate.call_count == 2  # before & after epoch 3

    # 7. resuming from epoch i, start = i+1/None, interval =1: perform
    #    evaluation after each epoch.
    runner = _build_demo_runner()
    evalhook = EvalHookParam(dataloader, start=2)
    evalhook.evaluate = MagicMock()
    runner.register_hook(evalhook)
    runner._epoch = 1
    runner.run([dataloader], [('train', 1)], 3)
    assert evalhook.evaluate.call_count == 2  # after epoch 2 & 3


def _build_demo_runner():

    class Model(nn.Module):

        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 1)

        def forward(self, x):
            return self.linear(x)

        def train_step(self, x, optimizer, **kwargs):
            return dict(loss=self(x))

        def val_step(self, x, optimizer, **kwargs):
            return dict(loss=self(x))

    model = Model()
    tmp_dir = tempfile.mkdtemp()

    runner = EpochBasedRunner(
        model=model, work_dir=tmp_dir, logger=logging.getLogger())
    return runner
