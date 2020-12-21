import math

import numpy as np
from mmcv.runner.hooks import HOOKS, LrUpdaterHook


def step_lr_interval(start_lr_factor, end_lr_factor, start_epoch, end_epoch):
    epochs = end_epoch - start_epoch
    mult = math.pow(end_lr_factor / start_lr_factor, 1. / (epochs))
    lr_intervals = start_lr_factor * (mult**np.arange(epochs))
    return lr_intervals


def log_lr_interval(start_lr_factor, end_lr_factor, start_epoch, end_epoch):
    epochs = end_epoch - start_epoch
    lr_intervals = np.logspace(
        math.log10(start_lr_factor), math.log10(end_lr_factor), epochs)
    return lr_intervals


@HOOKS.register_module()
class SiameseRPNLrUpdaterHook(LrUpdaterHook):

    lr_types = {'step': step_lr_interval, 'log': log_lr_interval}

    def __init__(self,
                 lr_configs=[
                     dict(
                         type='step',
                         start_lr_factor=0.2,
                         end_lr_factor=1.0,
                         end_epoch=5),
                     dict(
                         type='log',
                         start_lr_factor=1.0,
                         end_lr_factor=0.1,
                         end_epoch=20),
                 ],
                 **kwargs):
        super(SiameseRPNLrUpdaterHook, self).__init__(**kwargs)
        assert self.by_epoch is True
        self.lr_intervals = []

        start_epoch = 0
        for lr_config in lr_configs:
            lr_type = self.lr_types[lr_config.pop('type')]
            lr_config['start_epoch'] = start_epoch

            lr_intervals = lr_type(**lr_config)

            self.lr_intervals.append(lr_intervals)
            start_epoch = lr_config['end_epoch']
        self.lr_intervals = np.concatenate(self.lr_intervals)

    def get_lr(self, runner, base_lr):
        return base_lr * self.lr_intervals[runner.epoch]
