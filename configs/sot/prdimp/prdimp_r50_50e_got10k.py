_base_ = ['../../_base_/datasets/got10k.py', './prdimp_r50_50e_base.py']

# evaluator
val_evaluator = dict(outfile_prefix='results/prdimp/prdimp_got10k')
test_evaluator = val_evaluator
