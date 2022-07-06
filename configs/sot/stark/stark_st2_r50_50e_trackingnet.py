_base_ = [
    '../../_base_/datasets/trackingnet.py', './stark_st2_r50_50e_base.py'
]

# model setting
model = dict(test_cfg=dict(update_intervals=[25]))

# evaluator
val_evaluator = dict(outfile_prefix='results/stark_st2_trackingnet')
test_evaluator = val_evaluator
