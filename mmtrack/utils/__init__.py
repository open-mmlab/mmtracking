# Copyright (c) OpenMMLab. All rights reserved.
from .benchmark import (DataLoaderBenchmark, DatasetBenchmark,
                        InferenceBenchmark)
from .collect_env import collect_env
from .setup_env import register_all_modules

__all__ = [
    'collect_env', 'register_all_modules', 'DataLoaderBenchmark',
    'DatasetBenchmark', 'InferenceBenchmark'
]
