_base_ = ['./siamese_rpn_r50_20e_lasot.py']

# model settings
model = dict(
    test_cfg=dict(rpn=dict(penalty_k=0.1, window_influence=0.1, lr=0.5)))

data_root = 'data/'
# dataset settings
data = dict(
    val=dict(
        type='UAV123Dataset',
        ann_file=data_root + 'UAV123/annotations/uav123_infos.txt',
        img_prefix=data_root + 'UAV123',
        only_eval_visible=False),
    test=dict(
        type='UAV123Dataset',
        ann_file=data_root + 'UAV123/annotations/uav123_infos.txt',
        img_prefix=data_root + 'UAV123',
        only_eval_visible=False))
