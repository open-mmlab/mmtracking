_base_ = ['./siamese_rpn_r50_lasot.py']

data_root = 'data/'
# dataset settings
data = dict(
    test=dict(type='TrackingNetDataset', img_prefix=data_root + 'trackingnet'))
