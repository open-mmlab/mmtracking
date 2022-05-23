# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmdet.core.bbox.demodata import random_boxes

from mmtrack.datasets.transforms import (CropLikeSiamFC, SeqBlurAug,
                                         SeqColorAug, SeqShiftScaleAug)


class TestCropLikeSiamFC:

    def setup_class(cls):
        cls.crop_like_siamfc = CropLikeSiamFC(
            context_amount=0.5, exemplar_size=127, crop_size=255)
        cls.results = dict(
            img=np.random.randn(500, 500, 3),
            gt_bboxes=random_boxes(1, 256).numpy(),
            img_shape=(500, 500, 3))

    def test_transform(self):
        results = self.crop_like_siamfc(self.results)
        assert results['img'].shape == (255, 255, 3)
        assert results['gt_bboxes'].shape == (1, 4)
        assert results['img_shape'] == (255, 255, 3)


class TestSeqShiftScaleAug:

    def setup_class(cls):
        cls.seq_shift_scale_aug = SeqShiftScaleAug(
            target_size=[127, 255], shift=[4, 64], scale=[0.05, 0.18])
        img = np.random.randn(500, 500, 3)
        gt_bbox = random_boxes(1, 256).numpy()
        cls.results = dict(
            img=[img.copy(), img.copy()],
            gt_bboxes=[gt_bbox.copy(), gt_bbox.copy()],
            img_shape=[(500, 500, 3), (500, 500, 3)])

    def test_transform(self):
        results = self.seq_shift_scale_aug(self.results)
        assert results['img'][0].shape == (127, 127, 3)
        assert results['img'][1].shape == (255, 255, 3)
        assert results['gt_bboxes'][0].shape == (1, 4)
        assert results['gt_bboxes'][1].shape == (1, 4)
        assert results['img_shape'][0] == (127, 127, 3)
        assert results['img_shape'][1] == (255, 255, 3)


class TestSeqColorAug:

    def setup_class(cls):
        cls.seq_color_aug = SeqColorAug(prob=[1.0, 0.5])
        cls.results = dict(
            img=[np.random.randn(127, 127, 3),
                 np.random.randn(255, 255, 3)])

    def test_transform(self):
        results = self.seq_color_aug(self.results)
        assert results['img'][0].shape == (127, 127, 3)
        assert results['img'][1].shape == (255, 255, 3)


class TestSeqBlurAug:

    def setup_class(cls):
        cls.seq_blur_aug = SeqBlurAug(prob=[0.2, 0.5])
        cls.results = dict(
            img=[np.random.randn(127, 127, 3),
                 np.random.randn(255, 255, 3)])

    def test_transform(self):
        results = self.seq_blur_aug(self.results)
        assert results['img'][0].shape == (127, 127, 3)
        assert results['img'][1].shape == (255, 255, 3)
