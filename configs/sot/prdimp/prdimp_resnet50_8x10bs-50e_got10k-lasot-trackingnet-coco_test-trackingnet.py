_base_ = [
    '../../_base_/datasets/trackingnet.py',
    './prdimp_resnet50_8x10bs-50e_got10k-lasot-trackingnet-coco_base.py'
]

# evaluator
val_evaluator = dict(outfile_prefix='results/prdimp_trackingnet')
test_evaluator = val_evaluator
