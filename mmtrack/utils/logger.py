# Copyright (c) OpenMMLab. All rights reserved.
import logging

from mmcv.utils import get_logger


def get_root_logger(log_file=None, log_level=logging.INFO):
    """Get root logger.

    Args:
        log_file (str): File path of log. Defaults to None.
        log_level (int): The level of logger. Defaults to logging.INFO.

    Returns:
        :obj:`logging.Logger`: The obtained logger
    """
    return get_logger('mmtrack', log_file, log_level)
