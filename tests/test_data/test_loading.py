import copy
import os.path as osp

import numpy as np

from mmtrack.datasets.pipelines import LoadMultiImagesFromFile


class TestLoading(object):

    @classmethod
    def setup_class(cls):
        cls.data_prefix = osp.join(osp.dirname(__file__), '../assets')

    def test_load_seq_imgs(self):
        img_names = ['image_1.jpg', 'image_2.jpg', 'image_3.jpg']
        results = [
            dict(img_prefix=self.data_prefix, img_info=dict(filename=name))
            for name in img_names
        ]
        transform = LoadMultiImagesFromFile()
        all_results = transform(copy.deepcopy(results))
        assert isinstance(all_results, list)
        for i, results in enumerate(all_results):
            assert results['filename'] == osp.join(self.data_prefix,
                                                   img_names[i])
            assert results['ori_filename'] == img_names[i]
            assert results['img'].shape == (256, 512, 3)
            assert results['img'].dtype == np.uint8
            assert results['img_shape'] == (256, 512, 3)
            assert results['ori_shape'] == (256, 512, 3)
            assert repr(transform) == transform.__class__.__name__ + \
                "(to_float32=False, color_type='color', " + \
                "file_client_args={'backend': 'disk'})"
