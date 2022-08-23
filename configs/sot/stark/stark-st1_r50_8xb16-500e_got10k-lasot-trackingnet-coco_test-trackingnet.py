_base_ = [
    '../../_base_/datasets/trackingnet.py',
    './stark-st1_r50_8xb16-500e_got10k-lasot-trackingnet-coco_base.py'
]

# evaluator
val_evaluator = dict(outfile_prefix='results/stark_st1_trackingnet')
test_evaluator = val_evaluator
