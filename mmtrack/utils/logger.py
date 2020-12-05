import logging
from mmcv.utils import get_logger


def get_root_logger(log_file=None, log_level=logging.INFO):
    return get_logger('mmtrack', log_file, log_level)
