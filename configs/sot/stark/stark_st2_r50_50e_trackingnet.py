_base_ = ['./stark_st2_r50_50e_lasot.py']

# model setting
model = dict(test_cfg=dict(update_intervals=[25]))

data_root = 'data/'
data = dict(
    test=dict(type='TrackingNetDataset', img_prefix=data_root + 'trackingnet'))
