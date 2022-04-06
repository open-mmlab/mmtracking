_base_ = ['./stark_st1_r50_500e_lasot.py']

data_root = 'data/'
data = dict(
    test=dict(
        type='TrackingNetDataset',
        ann_file=data_root +
        'trackingnet/annotations/trackingnet_test_infos.txt',
        img_prefix=data_root + 'trackingnet'))
