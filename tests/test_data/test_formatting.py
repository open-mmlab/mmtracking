import os.path as osp

import numpy as np
from mmcv.utils import build_from_cfg

from mmtrack.datasets import PIPELINES


class TestFormatting(object):

    @classmethod
    def setup_class(cls):
        cls.data_prefix = osp.join(osp.dirname(__file__), '../assets')

    def test_formatting(self):
        img_names = ['image_1.jpg', 'image_2.jpg', 'image_3.jpg']
        collect_keys = ['img', 'gt_bboxes']
        num_objects = 4
        num_ref_imgs = len(img_names) - 1

        results = [
            dict(img_prefix=self.data_prefix, img_info=dict(filename=name))
            for name in img_names
        ]

        load = dict(type='LoadMultiImagesFromFile')
        load = build_from_cfg(load, PIPELINES)
        results = load(results)
        assert len(results) == len(img_names)

        for result in results:
            result['gt_bboxes'] = np.random.randn(num_objects, 4)

        collect = dict(type='VideoCollect', keys=collect_keys)
        collect = build_from_cfg(collect, PIPELINES)
        results = collect(results)
        assert len(results) == len(img_names)
        for key in collect_keys:
            assert key in results[0]
            assert key in results[1]
            assert key in results[2]
        assert 'img_metas' in results[0]
        assert 'img_metas' in results[1]
        assert 'img_metas' in results[2]
        key_results = results[0]

        concat_ref = dict(type='ConcatVideoReferences')
        concat_ref = build_from_cfg(concat_ref, PIPELINES)
        results = concat_ref(results)
        assert len(results) == 2
        assert results[0] == key_results
        assert results[1]['img'].ndim == 4
        assert results[1]['img'].shape[3] == 2
        assert len(results[1]['img_metas']) == 2
        assert results[1]['gt_bboxes'].ndim == 2
        assert results[1]['gt_bboxes'].shape[1] == 5
        assert results[1]['gt_bboxes'].shape[0] == (num_ref_imgs * num_objects)

        ref_prefix = 'ref'
        bundle = dict(type='SeqDefaultFormatBundle', ref_prefix=ref_prefix)
        bundle = build_from_cfg(bundle, PIPELINES)
        results = bundle(results)
        for key in results:
            if ref_prefix not in key:
                assert f'{ref_prefix}_{key}' in results

        import pdb
        pdb.set_trace()
