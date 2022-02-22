# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from mmtrack.datasets import build_dataset

PREFIX = osp.join(osp.dirname(__file__), '../../data/demo_sot_data/')


def test_random_sample_concatdataset():
    train_cfg = dict(
        type='RandomSampleConcatDataset',
        dataset_sampling_weights=[1, 1],
        dataset_cfgs=[
            dict(
                type='GOT10kDataset',
                img_prefix=PREFIX + 'got10k',
                pipeline=[],
                split='train',
                test_mode=False),
            dict(
                type='TrackingNetDataset',
                chunks_list=[0],
                img_prefix=PREFIX + 'trackingnet',
                pipeline=[],
                split='train',
                test_mode=False)
        ])
    dataset = build_dataset(train_cfg)
    results = dataset[0]
    assert len(dataset) == 4
    assert dataset.dataset_sampling_probs == [0.5, 0.5]
    assert len(results) == 2
