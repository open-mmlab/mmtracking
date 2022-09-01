_base_ = './mask2former_r50_8xb2-8e_youtubevis2019.py'

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')),
    init_cfg=dict(
        type='Pretrained',
        checkpoint='https://download.openmmlab.com/mmdetection/v2.0/'
        'mask2former/mask2former_r101_lsj_8x2_50e_coco/'
        'mask2former_r101_lsj_8x2_50e_coco_20220426_100250-c50b6fa6.pth'))
