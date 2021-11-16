_base_ = ['./masktrack_rcnn_r50_fpn_12e_youtubevis2019.py']
model = dict(
    detector=dict(
        backbone=dict(
            depth=101,
            init_cfg=dict(
                type='Pretrained', checkpoint='torchvision://resnet101')),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r101_fpn_1x_coco/mask_rcnn_r101_fpn_1x_coco_20200204-1efe0ed5.pth'  # noqa: E501
        )))

data_root = 'data/youtube_vis_2021/'
dataset_version = data_root[-5:-1]
data = dict(
    train=dict(
        dataset_version=dataset_version,
        ann_file=data_root + 'annotations/youtube_vis_2021_train.json',
        img_prefix=data_root + 'train/JPEGImages'),
    val=dict(
        dataset_version=dataset_version,
        ann_file=data_root + 'annotations/youtube_vis_2021_valid.json',
        img_prefix=data_root + 'valid/JPEGImages'),
    test=dict(
        dataset_version=dataset_version,
        ann_file=data_root + 'annotations/youtube_vis_2021_valid.json',
        img_prefix=data_root + 'valid/JPEGImages'))
