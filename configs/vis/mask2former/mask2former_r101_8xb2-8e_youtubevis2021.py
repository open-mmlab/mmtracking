_base_ = './mask2former_r50_8xb2-8e_youtubevis2021.py'

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')),
    init_cfg=dict(
        type='Pretrained',
        checkpoint='https://download.openmmlab.com/mmdetection/v3.0/'
        'mask2former/mask2former_r101_8xb2-lsj-50e_coco-panoptic'
        '/mask2former_r101_8xb2-lsj-50e_'
        'coco-panoptic_20220329_225104-c74d4d71.pth'))
