import numpy as np
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import Collect, to_tensor


@PIPELINES.register_module()
class ConcatVideoReferences(object):

    def __call__(self, results):
        assert (isinstance(results, list)), 'results must be list'
        outs = results[:1]
        for i, result in enumerate(results[1:], 1):
            if 'img' in result:
                img = result['img']
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                if i == 1:
                    result['img'] = np.expand_dims(img, -1)
                else:
                    outs[1]['img'] = np.concatenate(
                        (outs[1]['img'], np.expand_dims(img, -1)), axis=-1)
            for key in ['filename', 'ori_filename', 'gt_masks']:
                if key in result:
                    if i == 1:
                        result[key] = [result[key]]
                    else:
                        outs[1][key].append(result[key])
            for key in [
                    'proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels',
                    'gt_instance_ids'
            ]:
                if key not in result:
                    continue
                value = result[key]
                if value.ndim == 1:
                    value = value[:, None]
                N = value.shape[0]
                value = np.concatenate((np.full((N, 1), i - 1), value), axis=1)
                if i == 1:
                    result[key] = value
                else:
                    outs[1][key] = np.concatenate((outs[1][key], value),
                                                  axis=0)
            if 'gt_semantic_seg' in result:
                if i == 1:
                    result['gt_semantic_seg'] = result['gt_semantic_seg'][...,
                                                                          None,
                                                                          None]
                else:
                    outs[1]['gt_semantic_seg'] = np.concatenate(
                        (outs[1]['gt_semantic_seg'],
                         result['gt_semantic_seg'][..., None, None]),
                        axis=-1)
            if i == 1:
                outs.append(result)
        return outs


@PIPELINES.register_module()
class SeqDefaultFormatBundle(object):

    def __call__(self, results):
        outs = []
        for _results in results:
            _results = self.default_format_bundle(_results)
            outs.append(_results)
        return outs

    def default_format_bundle(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with \
                default bundle.
        """

        if 'img' in results:
            img = results['img']
            # add default meta keys
            results = self._add_default_meta_keys(results)
            if len(img.shape) == 3:
                img = np.ascontiguousarray(img.transpose(2, 0, 1))
            else:
                img = np.ascontiguousarray(img.transpose(3, 2, 0, 1))
            results['img'] = DC(to_tensor(img), stack=True)
        for key in [
                'proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels',
                'gt_instance_ids'
        ]:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]))
        if 'gt_masks' in results:
            results['gt_masks'] = DC(results['gt_masks'], cpu_only=True)
        if 'gt_semantic_seg' in results:
            semantic_seg = results['gt_semantic_seg']
            if len(semantic_seg.shape) == 2:
                semantic_seg = semantic_seg[None, ...]
            else:
                semantic_seg = np.ascontiguousarray(
                    semantic_seg.transpose(3, 2, 0, 1))
            results['gt_semantic_seg'] = DC(
                to_tensor(results['gt_semantic_seg']), stack=True)
        return results

    def _add_default_meta_keys(self, results):
        """Add default meta keys.

        We set default meta keys including `pad_shape`, `scale_factor` and
        `img_norm_cfg` to avoid the case where no `Resize`, `Normalize` and
        `Pad` are implemented during the whole pipeline.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            results (dict): Updated result dict contains the data to convert.
        """
        img = results['img']
        results.setdefault('pad_shape', img.shape)
        results.setdefault('scale_factor', 1.0)
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results.setdefault(
            'img_norm_cfg',
            dict(
                mean=np.zeros(num_channels, dtype=np.float32),
                std=np.ones(num_channels, dtype=np.float32),
                to_rgb=False))
        return results

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module(force=True)
class VideoCollect(Collect):

    def __init__(
            self,
            keys,
            ref_prefix='ref',
            default_meta_keys={
                'filename', 'ori_filename', 'ori_shape', 'img_shape',
                'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                'img_norm_cfg'
            },
            meta_keys='frame_id'):
        self.keys = keys
        self.ref_prefix = ref_prefix
        if isinstance(meta_keys, str):
            meta_keys = {meta_keys}
        elif isinstance(meta_keys, list):
            meta_keys = set(meta_keys)
        else:
            raise TypeError('meta_keys must be str or list')
        default_meta_keys.update(meta_keys)
        self.meta_keys = default_meta_keys

    def __call__(self, results):
        if self.ref_prefix is None:
            assert isinstance(results, dict), \
                'results must be a dict when self.ref_prefix is None'
            return super().__call__(results)

        assert len(results) == 2

        outs = []
        for _results in results:
            _results = super().__call__(_results)
            outs.append(_results)

        data = {}
        data.update(outs[0])
        for k, v in outs[1].items():
            data[f'{self.ref_prefix}_{k}'] = v

        return data
