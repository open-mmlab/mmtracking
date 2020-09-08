from mmcv.parallel import DataContainer as DC
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import Collect, DefaultFormatBundle, to_tensor


@PIPELINES.register_module()
class SeqDefaultFormatBundle(DefaultFormatBundle):

    def __call__(self, results):
        outs = []
        for _results in results:
            _results = super().__call__(_results)
            _results['gt_match_indices'] = DC(
                to_tensor(_results['gt_match_indices']))
            outs.append(_results)
        return outs


@PIPELINES.register_module()
class VideoCollect(Collect):
    """Collect data from the loader relevant to the specific task.

    This is usually the last stage of the data loader pipeline. Typically keys
    is set to some subset of "img", "proposals", "gt_bboxes",
    "gt_bboxes_ignore", "gt_labels", and/or "gt_masks".

    The "img_meta" item is always populated.  The contents of the "img_meta"
    dictionary depends on "meta_keys". By default this includes:

        - "img_shape": shape of the image input to the network as a tuple \
            (h, w, c).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.

        - "scale_factor": a float indicating the preprocessing scale

        - "flip": a boolean indicating if image flip transform was used

        - "filename": path to the image file

        - "ori_shape": original shape of the image as a tuple (h, w, c)

        - "pad_shape": image shape after padding

        - "img_norm_cfg": a dict of normalization information:

            - mean - per channel mean subtraction
            - std - per channel std divisor
            - to_rgb - bool indicating if bgr was converted to rgb

    Args:
        keys (Sequence[str]): Keys of results to be collected in ``data``.
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Default: ``('filename', 'ori_filename', 'ori_shape', 'img_shape',
            'pad_shape', 'scale_factor', 'flip', 'flip_direction',
            'img_norm_cfg')``
    """

    def __init__(self,
                 keys,
                 meta_keys=('filename', 'ori_filename', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'img_norm_cfg', 'frame_id')):
        self.keys = keys
        self.meta_keys = meta_keys


@PIPELINES.register_module(force=True)
class SeqCollect(VideoCollect):

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
