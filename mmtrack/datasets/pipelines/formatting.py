from mmcv.parallel import DataContainer as DC
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import Collect, DefaultFormatBundle, to_tensor


@PIPELINES.register_module()
class SeqDefaultFormatBundle(DefaultFormatBundle):

    def __call__(self, results):
        outs = []
        for _results in results:
            _results = super().__call__(_results)
            _results['gt_mids'] = DC(to_tensor(_results['gt_mids']))
            outs.append(_results)
        return outs


@PIPELINES.register_module()
class SeqCollect(Collect):

    def __init__(self,
                 keys,
                 ref_prefix='ref',
                 meta_keys=('filename', 'ori_filename', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'img_norm_cfg')):
        self.keys = keys
        self.ref_prefix = ref_prefix
        self.meta_keys = meta_keys

    def __call__(self, results):
        outs = []
        for _results in results:
            _results = super().__call__(_results)
            outs.append(_results)

        assert len(outs) == 2
        data = {}
        data.update(outs[0])
        for k, v in outs[1].items():
            data[f'{self.ref_prefix}_{k}'] = v

        return data
