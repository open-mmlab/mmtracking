_base_ = [
    '../../_base_/datasets/trackingnet.py', './siamese_rpn_r50_20e_base.py'
]

# evaluator
val_evaluator = dict(outfile_prefix='results/siamese_rpn_trackingnet')
test_evaluator = val_evaluator
