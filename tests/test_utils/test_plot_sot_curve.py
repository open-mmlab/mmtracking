# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmtrack.utils import (plot_norm_precision_curve, plot_precision_curve,
                           plot_success_curve)


def test_plot_success_curve():
    success_1 = np.arange(100, -1, -5) + (np.random.rand(21) - 0.5) * 4
    success_2 = np.arange(100, -1, -5) + (np.random.rand(21) - 0.5) * 4
    success = np.stack([success_1, success_2])
    plot_success_curve(success, ['tracker-1', 'tracker-2'])


def test_plot_norm_precision_curve():
    precision_1 = np.arange(0, 101, 2) + (np.random.rand(51) - 0.5) * 4
    precision_2 = np.arange(0, 101, 2) + (np.random.rand(51) - 0.5) * 4
    precision = np.stack([precision_1, precision_2])
    plot_norm_precision_curve(precision, ['tracker-1', 'tracker-2'])


def test_plot_precision_curve():
    precision_1 = np.arange(0, 101, 2) + (np.random.rand(51) - 0.5) * 4
    precision_2 = np.arange(0, 101, 2) + (np.random.rand(51) - 0.5) * 4
    precision = np.stack([precision_1, precision_2])
    plot_precision_curve(precision, ['tracker-1', 'tracker-2'])
