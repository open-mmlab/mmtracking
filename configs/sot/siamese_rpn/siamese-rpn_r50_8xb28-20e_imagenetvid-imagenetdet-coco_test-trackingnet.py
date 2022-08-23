_base_ = [
    '../../_base_/datasets/trackingnet.py',
    './siamese-rpn_r50_8xb28-20e_imagenetvid-imagenetdet-coco_base.py'
]

# evaluator
val_evaluator = dict(outfile_prefix='results/siamese_rpn_trackingnet')
test_evaluator = val_evaluator
