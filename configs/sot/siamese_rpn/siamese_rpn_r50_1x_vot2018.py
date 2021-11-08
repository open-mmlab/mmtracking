_base_ = ['./siamese_rpn_r50_1x_lasot.py']

# model settings
model = dict(
    test_cfg=dict(
        rpn=dict(penalty_k=0.04, window_influence=0.44, lr=0.33),
        test_mode='VOT'))

data_root = 'data/'
# dataset settings
data = dict(
    val=dict(
        type='VOTDataset',
        ann_file=data_root + 'vot2018/annotations/vot2018.json',
        img_prefix=data_root + 'vot2018/data'),
    test=dict(
        type='VOTDataset',
        ann_file=data_root + 'vot2018/annotations/vot2018.json',
        img_prefix=data_root + 'vot2018/data'))
evaluation = dict(
    metric=['track'], interval=1, start=10, rule='greater', save_best='eao')
