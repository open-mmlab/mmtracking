_base_ = [
    '../../_base_/datasets/trackingnet.py',
    './stark-st2_r50_8xb16-50e_got10k-lasot-trackingnet-coco_base.py'
]

# model setting
model = dict(test_cfg=dict(update_intervals=[25]))

# evaluator
val_evaluator = dict(outfile_prefix='results/stark_st2_trackingnet')
test_evaluator = val_evaluator
