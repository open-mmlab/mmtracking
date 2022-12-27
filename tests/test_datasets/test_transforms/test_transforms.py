# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmtrack.datasets.transforms import (BrightnessAug, CropLikeDiMP,
                                         CropLikeSiamFC, GrayAug, RandomCrop,
                                         SeqBboxJitter, SeqBlurAug,
                                         SeqColorAug, SeqCropLikeStark,
                                         SeqShiftScaleAug)
from mmtrack.testing import random_boxes


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


class TestGrayAug:

    def setup_class(cls):
        cls.gray_aug = GrayAug(prob=1)
        cls.results = dict(
            img=np.random.randint(0, 255, (127, 127, 3)).astype(np.uint8))

    def test_transform(self):
        results = self.gray_aug(self.results)
        assert results['img'].shape == (127, 127, 3)


class TestBrightnessAug:

    def setup_class(cls):
        cls.brightness_aug = BrightnessAug(jitter_range=0.2)
        cls.results = dict(img=np.random.randn(127, 127, 3))

    def test_transform(self):
        results = self.brightness_aug(self.results)
        assert results['img'].shape == (127, 127, 3)


class TestSeqBboxJitter:

    def setup_class(cls):
        cls.seq_shift_scale_aug = SeqBboxJitter(
            center_jitter_factor=[0, 4.5],
            scale_jitter_factor=[0, 0.5],
            crop_size_factor=[2, 5])
        gt_bbox = random_boxes(1, 256).numpy()
        cls.results = dict(gt_bboxes=[gt_bbox.copy(), gt_bbox.copy()])

    def test_transform(self):
        results = self.seq_shift_scale_aug(self.results)
        assert results['jittered_bboxes'][0].shape == (1, 4)
        assert results['jittered_bboxes'][1].shape == (1, 4)


class TestSeqCropLikeStark:

    def setup_class(cls):
        cls.seq_crop_like_stark = SeqCropLikeStark(
            crop_size_factor=[2, 5], output_size=[128, 320])
        cls.results = dict(
            img=[np.random.randn(500, 500, 3),
                 np.random.randn(500, 500, 3)],
            gt_bboxes=[
                random_boxes(1, 256).numpy(),
                random_boxes(1, 256).numpy()
            ],
            img_shape=[(500, 500, 3), (500, 500, 3)],
            jittered_bboxes=[
                random_boxes(1, 256).numpy(),
                random_boxes(1, 256).numpy()
            ])

    def test_transform(self):
        results = self.seq_crop_like_stark(self.results)
        assert results['img'][0].shape == (128, 128, 3)
        assert results['img'][1].shape == (320, 320, 3)
        assert results['gt_bboxes'][0].shape == (1, 4)
        assert results['gt_bboxes'][1].shape == (1, 4)
        assert results['img_shape'][0] == (128, 128, 3)
        assert results['img_shape'][1] == (320, 320, 3)
        assert results['padding_mask'][0].shape == (128, 128)
        assert results['padding_mask'][1].shape == (320, 320)


class TestCropLikeDiMP:

    def setup_class(cls):
        cls.crop_like_dimp = CropLikeDiMP(crop_size_factor=5, output_size=255)
        cls.results = dict(
            img=np.random.randn(500, 500, 3),
            gt_bboxes=random_boxes(1, 100).numpy(),
            img_shape=(500, 500, 3),
            jittered_bboxes=random_boxes(1, 100).numpy())

    def test_transform(self):
        results = self.crop_like_dimp(self.results)
        assert results['img'].shape == (255, 255, 3)
        assert results['gt_bboxes'].shape == (1, 4)
        assert results['img_shape'] == (255, 255, 3)


class TestRandomCrop:

    def setup_class(cls):
        cls.random_crop = RandomCrop(
            crop_size=(256, 256), allow_negative_crop=True)
        cls.results = dict(
            img=np.random.randn(512, 512, 3),
            gt_bboxes=random_boxes(10, 100).numpy(),
            img_shape=(512, 512, 3),
            gt_instances_id=np.array(list(range(10))))

    def test_transform(self):
        results = self.random_crop(self.results)
        assert results['img'].shape == (256, 256, 3)
        assert results['img_shape'] == (256, 256, 3)
        # Ensure that the instance_id is also filtered out correctly
        assert results['gt_bboxes'].shape[0] == results[
            'gt_instances_id'].shape[0]
