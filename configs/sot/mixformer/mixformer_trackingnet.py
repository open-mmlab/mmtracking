_base_ = ['./mixformer_got10k.py']

data_root = 'data/'
data = dict(
    test=dict(
        type='TrackingNetDataset',
        ann_file=data_root + 'trackingnet/annotations/trackingnet_test_infos.txt',
        img_prefix=data_root + 'trackingnet'))