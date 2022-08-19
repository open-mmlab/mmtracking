_base_ = [
    '../../_base_/datasets/vot2018.py',
    './siamese-rpn_resnet50_8x28bs-20e_imagenetvid-imagenetdet-coco_base.py'
]

# model settings
model = dict(
    test_cfg=dict(
        rpn=dict(penalty_k=0.04, window_influence=0.44, lr=0.33),
        test_mode='VOT'))
