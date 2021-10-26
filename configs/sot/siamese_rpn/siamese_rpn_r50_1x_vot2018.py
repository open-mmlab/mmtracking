_base_ = ['./siamese_rpn_r50_1x_lasot.py']

# model settings
model = dict(
    test_cfg=dict(
        rpn=dict(penalty_k=0.16, window_influence=0.4, lr=0.35),
        criteria='VOT'))

data_root = 'data/'
# dataset settings
data = dict(
    val=dict(
        type='VOT2018Dataset',
        ann_file=data_root + 'vot2018/annotations/vot2018.json',
        img_prefix=data_root + 'vot2018/data'),
    test=dict(
        type='VOT2018Dataset',
        ann_file=data_root + 'vot2018/annotations/vot2018.json',
        img_prefix=data_root + 'vot2018/data'))
evaluation = dict(
    metric=['track'], interval=1, start=10, rule='greater', save_best='eao')
