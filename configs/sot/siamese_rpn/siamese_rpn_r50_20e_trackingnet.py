_base_ = ['./siamese_rpn_r50_20e_lasot.py']

# dataloader
val_dataloader = dict(
    dataset=dict(
        type='TrackingNetDataset',
        ann_file='trackingnet/annotations/trackingnet_test_infos.txt',
        data_prefix=dict(img_path='trackingnet')))
test_dataloader = val_dataloader

# evaluator
val_evaluator = dict(
    type='SOTMetric',
    format_only=True,
    outfile_prefix='results/submitted_trackingnet')
test_evaluator = val_evaluator
