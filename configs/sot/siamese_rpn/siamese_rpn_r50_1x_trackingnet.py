_base_ = ['./siamese_rpn_r50_1x_lasot.py']

# model settings
model = dict(
    test_cfg=dict(rpn=dict(penalty_k=0.05, window_influence=0.42, lr=0.38)))

data_root = 'data/'
# dataset settings
data = dict(
    val=dict(
        type='TrackingNetTestDataset',
        ann_file=data_root +
        'trackingnet/TEST/annotations/trackingnet_test.json',
        img_prefix=data_root + 'trackingnet/TEST/frames',
    ),
    test=dict(
        type='TrackingNetTestDataset',
        ann_file=data_root +
        'trackingnet/TEST/annotations/trackingnet_test.json',
        img_prefix=data_root + 'trackingnet/TEST/frames'))
