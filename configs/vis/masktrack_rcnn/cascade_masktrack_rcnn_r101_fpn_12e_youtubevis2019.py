_base_ = ['./cascade_masktrack_rcnn_r50_fpn_12e_youtubevis2019.py']
model = dict(
    detector=dict(
        backbone=dict(
            depth=101,
            init_cfg=dict(
                type='Pretrained', checkpoint='torchvision://resnet101')),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_r101_fpn_1x_coco/cascade_mask_rcnn_r101_fpn_1x_coco_20200203-befdf6ee.pth'  # noqa: E501
        )))
