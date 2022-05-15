# TODO: remove this wrapper after mmcv fix the bug about TransformBroadcaster
# Copyright (c) OpenMMLab. All rights reserved.
from collections.abc import Sequence
from typing import Callable, Dict, List, Optional, Union

from mmcv.transforms import KeyMapper
from mmcv.transforms.utils import cache_random_params

from mmtrack.registry import TRANSFORMS

# Import nullcontext if python>=3.7, otherwise use a simple alternative
# implementation.
try:
    from contextlib import nullcontext
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def nullcontext(resource=None):
        try:
            yield resource
        finally:
            pass


@TRANSFORMS.register_module()
class TransformBroadcaster(KeyMapper):
    """A transform wrapper to apply the wrapped transforms to multiple data
    items. For example, apply Resize to multiple images.

    Args:
        transforms (list[dict | callable]): Sequence of transform object or
            config dict to be wrapped.
        mapping (dict): A dict that defines the input key mapping.
            Note that to apply the transforms to multiple data items, the
            outer keys of the target items should be remapped as a list with
            the standard inner key (The key required by the wrapped transform).
            See the following example and the document of
            ``mmcv.transforms.wrappers.KeyMapper`` for details.
        remapping (dict): A dict that defines the output key mapping.
            The keys and values have the same meanings and rules as in the
            ``mapping``. Default: None.
        auto_remap (bool, optional): If True, an inverse of the mapping will
            be used as the remapping. If auto_remap is not given, it will be
            automatically set True if 'remapping' is not given, and vice
            versa. Default: None.
        allow_nonexist_keys (bool): If False, the outer keys in the mapping
            must exist in the input data, or an exception will be raised.
            Default: False.
        share_random_params (bool): If True, the random transform
            (e.g., RandomFlip) will be conducted in a deterministic way and
            have the same behavior on all data items. For example, to randomly
            flip either both input image and ground-truth image, or none.
            Default: False.

    .. note::
        To apply the transforms to each elements of a list or tuple, instead
        of separating data items, you can map the outer key of the target
        sequence to the standard inner key. See example 2.
        example.

    Examples:
        >>> # Example 1:
        >>> pipeline = [
        >>>     dict(type='LoadImageFromFile', key='lq'),  # low-quality img
        >>>     dict(type='LoadImageFromFile', key='gt'),  # ground-truth img
        >>>     # TransformBroadcaster maps multiple outer fields to standard
        >>>     # the inner field and process them with wrapped transforms
        >>>     # respectively
        >>>     dict(type='TransformBroadcaster',
        >>>         # case 1: from multiple outer fields
        >>>         mapping={'img': ['lq', 'gt']},
        >>>         auto_remap=True,
        >>>         # share_random_param=True means using identical random
        >>>         # parameters in every processing
        >>>         share_random_param=True,
        >>>         transforms=[
        >>>             dict(type='Crop', crop_size=(384, 384)),
        >>>             dict(type='Normalize'),
        >>>         ])
        >>> ]
        >>> # Example 2:
        >>> pipeline = [
        >>>     dict(type='LoadImageFromFile', key='lq'),  # low-quality img
        >>>     dict(type='LoadImageFromFile', key='gt'),  # ground-truth img
        >>>     # TransformBroadcaster maps multiple outer fields to standard
        >>>     # the inner field and process them with wrapped transforms
        >>>     # respectively
        >>>     dict(type='TransformBroadcaster',
        >>>         # case 2: from one outer field that contains multiple
        >>>         # data elements (e.g. a list)
        >>>         # mapping={'img': 'images'},
        >>>         auto_remap=True,
        >>>         share_random_param=True,
        >>>         transforms=[
        >>>             dict(type='Crop', crop_size=(384, 384)),
        >>>             dict(type='Normalize'),
        >>>         ])
        >>> ]
    """

    def __init__(self,
                 transforms: List[Union[Dict, Callable[[Dict], Dict]]],
                 mapping: Optional[Dict] = None,
                 remapping: Optional[Dict] = None,
                 auto_remap: Optional[bool] = None,
                 allow_nonexist_keys: bool = False,
                 share_random_params: bool = False):
        super().__init__(transforms, mapping, remapping, auto_remap,
                         allow_nonexist_keys)

        self.share_random_params = share_random_params

    def scatter_sequence(self, data: Dict) -> List[Dict]:
        # infer split number from input
        seq_len = None
        key_rep = None
        if self.mapping:
            keys = self.mapping.keys()
        else:
            keys = data.keys()

        for key in keys:
            assert isinstance(data[key], Sequence)
            if seq_len is not None:
                if len(data[key]) != seq_len:
                    raise ValueError('Got inconsistent sequence length: '
                                     f'{seq_len} ({key_rep}) vs. '
                                     f'{len(data[key])} ({key})')
            else:
                seq_len = len(data[key])
                key_rep = key

        scatters = []
        for i in range(seq_len):
            scatter = data.copy()
            for key in keys:
                scatter[key] = data[key][i]
            scatters.append(scatter)
        return scatters

    def transform(self, results: Dict):
        # Apply input remapping
        inputs = results
        if self.mapping:
            inputs = self.map_input(inputs, self.mapping)

        # Scatter sequential inputs into a list
        inputs = self.scatter_sequence(inputs)

        # Control random parameter sharing with a context manager
        if self.share_random_params:
            # The context manager :func`:cache_random_params` will let
            # cacheable method of the transforms cache their outputs. Thus
            # the random parameters will only generated once and shared
            # by all data items.
            ctx = cache_random_params
        else:
            ctx = nullcontext

        with ctx(self.transforms):
            outputs = [self.transforms(_input) for _input in inputs]

        # Collate output scatters (list of dict to dict of list)
        outputs = {
            key: [_output[key] for _output in outputs]
            for key in outputs[0]
        }

        # Apply output remapping
        if self.remapping:
            outputs = self.map_output(outputs, self.remapping)

        results.update(outputs)
        return results
