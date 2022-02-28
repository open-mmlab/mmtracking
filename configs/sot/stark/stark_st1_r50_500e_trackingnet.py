_base_ = ['./stark_st1_r50_500e_lasot.py']

data_root = 'data/'
data = dict(
    test=dict(type='TrackingNetDataset', img_prefix=data_root + 'trackingnet'))
