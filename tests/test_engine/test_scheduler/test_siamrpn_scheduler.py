# Copyright (c) OpenMMLab. All rights reserved.
import math
from unittest import TestCase

import torch
import torch.nn.functional as F
import torch.optim as optim
from mmengine.optim.scheduler import _ParamScheduler
from mmengine.testing import assert_allclose

from mmtrack.engine import SiamRPNExpLR, SiamRPNExpParamScheduler


class ToyModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 1)
        self.conv2 = torch.nn.Conv2d(1, 1, 1)

    def forward(self, x):
        return self.conv2(F.relu(self.conv1(x)))


class TestSiamRPNExpScheduler(TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        self.model = ToyModel()
        self.base_lr = 0.0005
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.base_lr,
            momentum=0.01,
            weight_decay=5e-4)

    def _test_scheduler_value(self,
                              schedulers,
                              targets,
                              epochs=5,
                              param_name='lr'):
        if isinstance(schedulers, _ParamScheduler):
            schedulers = [schedulers]
        for epoch in range(epochs):
            for param_group, target in zip(self.optimizer.param_groups,
                                           targets):
                print(param_group[param_name])
                assert_allclose(
                    target[epoch],
                    param_group[param_name],
                    msg='{} is wrong in epoch {}: expected {}, got {}'.format(
                        param_name, epoch, target[epoch],
                        param_group[param_name]),
                    atol=1e-5,
                    rtol=0)
            [scheduler.step() for scheduler in schedulers]

    def test_siamrpn_exp_scheduler(self):
        with self.assertRaises(ValueError):
            SiamRPNExpParamScheduler(self.optimizer, param_name='lr')
        epochs = 5
        start_factor = 0.2
        end_factor = 1.0
        mult = math.pow(end_factor / start_factor, 1. / (epochs))
        targets = [[
            self.base_lr * start_factor * (mult**i) for i in range(epochs)
        ]]

        scheduler = SiamRPNExpParamScheduler(
            self.optimizer,
            param_name='lr',
            start_factor=start_factor,
            end_factor=end_factor,
            end=epochs,
            endpoint=False)
        self._test_scheduler_value(scheduler, targets, epochs)

    def test_siamrpn_exp_scheduler_convert_iterbased(self):
        epochs = 5
        epoch_length = 10

        iters = epochs * epoch_length
        start_factor = 0.2
        end_factor = 1.0
        mult = math.pow(end_factor / start_factor, 1. / (iters))
        targets = [[
            self.base_lr * start_factor * (mult**i) for i in range(iters)
        ]]
        scheduler = SiamRPNExpParamScheduler.build_iter_from_epoch(
            self.optimizer,
            param_name='lr',
            start_factor=start_factor,
            end_factor=end_factor,
            end=epochs,
            endpoint=False,
            epoch_length=epoch_length)
        self._test_scheduler_value(scheduler, targets, iters)

    def test_siamrpn_exp_lr(self):
        epochs = 5
        start_factor = 0.2
        end_factor = 1.0
        mult = math.pow(end_factor / start_factor, 1. / (epochs))
        targets = [[
            self.base_lr * start_factor * (mult**i) for i in range(epochs)
        ]]

        scheduler = SiamRPNExpLR(
            self.optimizer,
            start_factor=start_factor,
            end_factor=end_factor,
            end=epochs,
            endpoint=False)
        self._test_scheduler_value(scheduler, targets, epochs)
