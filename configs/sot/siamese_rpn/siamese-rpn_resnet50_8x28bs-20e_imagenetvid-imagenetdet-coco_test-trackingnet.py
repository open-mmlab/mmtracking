_base_ = [
    '../../_base_/datasets/trackingnet.py',
    './siamese-rpn_resnet50_8x28bs-20e_imagenetvid-imagenetdet-coco_base.py'
]

# evaluator
val_evaluator = dict(outfile_prefix='results/siamese_rpn_trackingnet')
test_evaluator = val_evaluator
