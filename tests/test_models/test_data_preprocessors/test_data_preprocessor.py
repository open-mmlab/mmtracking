# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

from mmtrack.models.data_preprocessors import TrackDataPreprocessor
from mmtrack.testing import demo_mm_inputs


class TestTrackDataPreprocessor(TestCase):

    def test_init(self):
        # test mean is None
        processor = TrackDataPreprocessor()
        self.assertTrue(not hasattr(processor, 'mean'))
        self.assertTrue(processor._enable_normalize is False)

        # test mean is not None
        processor = TrackDataPreprocessor(mean=[0, 0, 0], std=[1, 1, 1])
        self.assertTrue(hasattr(processor, 'mean'))
        self.assertTrue(hasattr(processor, 'std'))
        self.assertTrue(processor._enable_normalize)

        # please specify both mean and std
        with self.assertRaises(AssertionError):
            TrackDataPreprocessor(mean=[0, 0, 0])

        # bgr2rgb and rgb2bgr cannot be set to True at the same time
        with self.assertRaises(AssertionError):
            TrackDataPreprocessor(bgr_to_rgb=True, rgb_to_bgr=True)

    def test_forward(self):
        processor = TrackDataPreprocessor(mean=[0, 0, 0], std=[1, 1, 1])

        data = demo_mm_inputs(
            batch_size=1,
            frame_id=0,
            num_key_imgs=1,
            ref_prefix='search',
            image_shapes=[(3, 11, 10)],
            num_items=[1])
        out_data = processor(data)
        inputs, data_samples = out_data['inputs'], out_data['data_samples']
        for _, inputs_single_mode in inputs.items():
            self.assertEqual(inputs_single_mode.shape, (1, 1, 3, 11, 10))
        self.assertEqual(len(data_samples), 1)

        # test channel_conversion
        processor = TrackDataPreprocessor(
            mean=[0., 0., 0.], std=[1., 1., 1.], bgr_to_rgb=True)
        out_data = processor(data)
        inputs, data_samples = out_data['inputs'], out_data['data_samples']
        for _, inputs_single_mode in inputs.items():
            self.assertEqual(inputs_single_mode.shape, (1, 1, 3, 11, 10))
        self.assertEqual(len(data_samples), 1)

        # test padding
        data = demo_mm_inputs(
            batch_size=2,
            frame_id=0,
            num_key_imgs=1,
            ref_prefix='search',
            image_shapes=[(3, 10, 11), (3, 9, 14)],
            num_items=[1, 1])
        out_data = processor(data)
        inputs, data_samples = out_data['inputs'], out_data['data_samples']
        for _, inputs_single_mode in inputs.items():
            self.assertEqual(inputs_single_mode.shape, (2, 1, 3, 10, 14))

        # test pad_size_divisor
        data = demo_mm_inputs(
            batch_size=2,
            frame_id=0,
            num_key_imgs=1,
            ref_prefix='search',
            image_shapes=[(3, 10, 11), (3, 9, 24)],
            num_items=[1, 1])
        processor = TrackDataPreprocessor(
            mean=[0., 0., 0.], std=[1., 1., 1.], pad_size_divisor=5)
        out_data = processor(data)
        inputs, data_samples = out_data['inputs'], out_data['data_samples']
        for _, inputs_single_mode in inputs.items():
            self.assertEqual(inputs_single_mode.shape, (2, 1, 3, 10, 25))
        self.assertEqual(len(data_samples), 2)
        for data_sample, expected_shape in zip(data_samples, [(10, 15),
                                                              (10, 25)]):
            self.assertEqual(data_sample.pad_shape, expected_shape)
            self.assertEqual(data_sample.search_pad_shape, expected_shape)

        # test pad_mask=True
        data = demo_mm_inputs(
            batch_size=2,
            frame_id=0,
            num_key_imgs=1,
            ref_prefix='search',
            image_shapes=[(3, 10, 11), (3, 9, 24)],
            num_items=[1, 1],
            with_mask=True)
        processor = TrackDataPreprocessor(pad_mask=True, mask_pad_value=0)
        mask_pad_sums = [
            x.gt_instances.masks.masks.sum() for x in data['data_samples']
        ]
        out_data = processor(data)
        inputs, data_samples = out_data['inputs'], out_data['data_samples']
        for data_sample, expected_shape, mask_pad_sum in zip(
                data_samples, [(10, 24), (10, 24)], mask_pad_sums):
            self.assertEqual(data_sample.gt_instances.masks.masks.shape[-2:],
                             expected_shape)
            self.assertEqual(data_sample.gt_instances.masks.masks.sum(),
                             mask_pad_sum)
            self.assertEqual(
                data_sample.search_gt_instances.masks.masks.shape[-2:],
                expected_shape)
            self.assertEqual(data_sample.search_gt_instances.masks.masks.sum(),
                             mask_pad_sum)
