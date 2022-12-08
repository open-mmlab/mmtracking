_base_ = [
    '../../../configs/sot/siamese_rpn/siamese-rpn_r50_8xb28-20e_imagenetvid-imagenetdet-coco_test-lasot.py'  # noqa: E501
]

custom_imports = dict(imports=['projects.example_project.dummy'])

_base_.model.backbone.type = 'DummyResNet'
