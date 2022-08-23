_base_ = [
    '../../_base_/datasets/got10k.py',
    './prdimp_r50_8xb10-50e_got10k-lasot-trackingnet-coco_base.py'
]

# evaluator
val_evaluator = dict(outfile_prefix='results/prdimp/prdimp_got10k')
test_evaluator = val_evaluator
