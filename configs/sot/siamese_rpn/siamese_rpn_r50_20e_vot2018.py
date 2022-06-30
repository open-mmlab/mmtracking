_base_ = ['./siamese_rpn_r50_20e_lasot.py']

# model settings
model = dict(
    test_cfg=dict(
        rpn=dict(penalty_k=0.04, window_influence=0.44, lr=0.33),
        test_mode='VOT'))

# dataloader
val_dataloader = dict(
    dataset=dict(
        type='VOTDataset',
        ann_file='vot2018/annotations/vot2018_infos.txt',
        data_prefix=dict(img_path='vot2018')))
test_dataloader = val_dataloader

# evaluator
val_evaluator = dict(
    type='SOTMetric',
    _delete_=True,
    metric='VOT',
    metric_options=dict(dataset_type='vot2018'))
test_evaluator = val_evaluator
