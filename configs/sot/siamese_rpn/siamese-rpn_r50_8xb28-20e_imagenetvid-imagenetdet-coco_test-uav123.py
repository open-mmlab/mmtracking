_base_ = [
    '../../_base_/datasets/uav123.py',
    './siamese-rpn_r50_8xb28-20e_imagenetvid-imagenetdet-coco_base.py'
]

# model settings
model = dict(
    test_cfg=dict(rpn=dict(penalty_k=0.1, window_influence=0.1, lr=0.5)))
