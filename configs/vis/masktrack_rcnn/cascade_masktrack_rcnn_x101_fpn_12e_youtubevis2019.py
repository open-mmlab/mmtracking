_base_ = ['./cascade_masktrack_rcnn_r50_fpn_12e_youtubevis2019.py']
model = dict(
    detector=dict(
        backbone=dict(
            type='ResNeXt',
            depth=101,
            groups=64,
            base_width=4,
            init_cfg=dict(
                type='Pretrained',
                checkpoint='open-mmlab://resnext101_64x4d')),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_x101_64x4d_fpn_1x_coco/cascade_mask_rcnn_x101_64x4d_fpn_1x_coco_20200203-9a2db89d.pth'  # noqa: E501
        )))
