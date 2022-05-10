# Copyright (c) OpenMMLab. All rights reserved.
import logging
import multiprocessing as mp
import os
import platform
import tempfile
import warnings

import cv2
import torch


def setup_multi_processes(cfg):
    # set multi-process start method as `fork` to speed up the training
    if platform.system() != 'Windows':
        mp_start_method = cfg.get('mp_start_method', 'fork')
        mp.set_start_method(mp_start_method)

    # disable opencv multithreading to avoid system being overloaded
    opencv_num_threads = cfg.get('opencv_num_threads', 0)
    cv2.setNumThreads(opencv_num_threads)

    # setup OMP threads
    # This code is referred from https://github.com/pytorch/pytorch/blob/master/torch/distributed/run.py  # noqa
    if ('OMP_NUM_THREADS' not in os.environ and cfg.data.workers_per_gpu > 1):
        omp_num_threads = 1
        warnings.warn(
            f'Setting OMP_NUM_THREADS environment variable for each process '
            f'to be {omp_num_threads} in default, to avoid your system being '
            f'overloaded, please further tune the variable for optimal '
            f'performance in your application as needed.')
        os.environ['OMP_NUM_THREADS'] = str(omp_num_threads)

    # setup MKL threads
    if 'MKL_NUM_THREADS' not in os.environ and cfg.data.workers_per_gpu > 1:
        mkl_num_threads = 1
        warnings.warn(
            f'Setting MKL_NUM_THREADS environment variable for each process '
            f'to be {mkl_num_threads} in default, to avoid your system being '
            f'overloaded, please further tune the variable for optimal '
            f'performance in your application as needed.')
        os.environ['MKL_NUM_THREADS'] = str(mkl_num_threads)


def init_model_weights_quiet(model):
    """Creating a temporary file to record the information of initialized
    parameters. If not, the information of initialized parameters will be
    printed to the console because of the call of
    `mmcv.runner.BaseModule.init_weights`.

    Args:
        model: The model inheriented from `BaseModule` in MMCV.
    """
    assert hasattr(model, 'logger') and hasattr(model, 'init_weights')
    tmp_file = tempfile.NamedTemporaryFile(delete=False)
    file_handler = logging.FileHandler(tmp_file.name, mode='w')
    model.logger.addHandler(file_handler)
    # We need call `init_weights()` to load pretained weights in MOT
    # task.
    model.init_weights()
    file_handler.close()
    model.logger.removeHandler(file_handler)
    tmp_file.close()
    os.remove(tmp_file.name)


def max2d(input):
    """Computes the value and position of maximum in the last two dimensions.

    Args:
        input (Tensor): of shape (..., H, W)

    Returns:
        max_val (Tensor): The maximum value.
        argmax (Tensor): The position of maximum in [row, col] format.
    """

    max_val_row, argmax_row = torch.max(input, dim=-2)
    max_val, argmax_col = torch.max(max_val_row, dim=-1)
    argmax_row = argmax_row.view(argmax_col.numel(),
                                 -1)[torch.arange(argmax_col.numel()),
                                     argmax_col.view(-1)]
    argmax_row = argmax_row.reshape(argmax_col.shape)
    argmax = torch.cat((argmax_row.unsqueeze(-1), argmax_col.unsqueeze(-1)),
                       -1)
    return max_val, argmax
