# Copyright (c) OpenMMLab. All rights reserved.
try:
    from vot.region import calculate_overlap
except ImportError:
    calculate_overlap = None


def calculate_region_overlap(*args, **kwargs):
    if calculate_overlap is None:
        raise ImportError(
            'Please run'
            'pip install git+https://github.com/votchallenge/toolkit.git'
            'to manually install vot-toolkit')
    return calculate_overlap(*args, **kwargs)
