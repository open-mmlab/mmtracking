_base_ = ['./siamese_rpn_r50_20e_lasot.py']

# model settings
model = dict(
    test_cfg=dict(rpn=dict(penalty_k=0.1, window_influence=0.1, lr=0.5)))

# dataloader
val_dataloader = dict(
    dataset=dict(
        type='UAV123Dataset',
        ann_file='UAV123/annotations/uav123_infos.txt',
        data_prefix=dict(img_path='UAV123')))
test_dataloader = val_dataloader

# evaluator
val_evaluator = dict(
    type='SOTMetric', metric_options=dict(only_eval_visible=False))
test_evaluator = val_evaluator
