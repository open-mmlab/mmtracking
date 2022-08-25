_base_ = [
    './faster-rcnn_resnet50-fpn_8xb2-4e_mot17halftrain_test-mot17halfval.py'
]
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')),
    init_cfg=dict(
        type='Pretrained',
        checkpoint=  # noqa: E251
        'http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r101_fpn_2x_coco/faster_rcnn_r101_fpn_2x_coco_bbox_mAP-0.398_20200504_210455-1d2dac9c.pth'  # noqa: E501
    ))
