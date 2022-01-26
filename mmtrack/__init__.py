# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import mmcv
import mmdet
from packaging.version import parse

from .version import __version__, version_info

MMCV_MIN = '1.3.17'
MMCV_MAX = '1.5.0'

MMDET_MIN = '2.19.1'


def digit_version(version_str: str, length: int = 4):
    """Convert a version string into a tuple of integers.

    This method is usually used for comparing two versions. For pre-release
    versions: alpha < beta < rc.

    Args:
        version_str (str): The version string.
        length (int): The maximum number of version levels. Default: 4.

    Returns:
        tuple[int]: The version info in digits (integers).
    """
    version = parse(version_str)
    assert version.release, f'failed to parse version {version_str}'
    release = list(version.release)
    release = release[:length]
    if len(release) < length:
        release = release + [0] * (length - len(release))
    if version.is_prerelease:
        mapping = {'a': -3, 'b': -2, 'rc': -1}
        val = -4
        # version.pre can be None
        if version.pre:
            if version.pre[0] not in mapping:
                warnings.warn(f'unknown prerelease version {version.pre[0]}, '
                              'version checking may go wrong')
            else:
                val = mapping[version.pre[0]]
            release.extend([val, version.pre[-1]])
        else:
            release.extend([val, 0])

    elif version.is_postrelease:
        release.extend([1, version.post])
    else:
        release.extend([0, 0])
    return tuple(release)


mmcv_min_version = digit_version(MMCV_MIN)
mmcv_max_version = digit_version(MMCV_MAX)
mmcv_version = digit_version(mmcv.__version__)


assert (mmcv_min_version <= mmcv_version <= mmcv_max_version), \
    f'MMCV=={mmcv.__version__} is used but incompatible. ' \
    f'Please install mmcv>={MMCV_MIN}, <={MMCV_MAX}.'

mmdet_min_version = digit_version(MMDET_MIN)
mmdet_version = digit_version(mmdet.__version__)


assert (mmdet_min_version <= mmdet_version), \
    f'MMDet=={mmdet.__version__} is used but incompatible. ' \
    f'Please install mmdet>={MMDET_MIN}.'

__all__ = ['__version__', 'version_info', 'digit_version']
