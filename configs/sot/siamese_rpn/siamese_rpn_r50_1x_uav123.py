_base_ = ['./siamese_rpn_r50_1x_lasot.py']

# model settings
model = dict(
    test_cfg=dict(rpn=dict(penalty_k=0.01, window_influence=0.02, lr=0.46)))

data_root = 'data/'
# dataset settings
data = dict(
    val=dict(
        type='UAV123Dataset',
        ann_file=data_root + 'UAV123/annotations/uav123.json',
        img_prefix=data_root + 'UAV123/data_seq/UAV123'),
    test=dict(
        type='UAV123Dataset',
        ann_file=data_root + 'UAV123/annotations/uav123.json',
        img_prefix=data_root + 'UAV123/data_seq/UAV123'))
