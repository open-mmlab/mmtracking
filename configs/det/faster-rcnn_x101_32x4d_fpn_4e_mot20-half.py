USE_MMDET = True
_base_ = ['./faster-rcnn_r50_fpn_4e_mot17-half.py']
model = dict(
    detector=dict(
        pretrained='open-mmlab://resnext101_32x4d',
        backbone=dict(
            type='ResNeXt',
            depth=101,
            groups=32,
            base_width=4,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            style='pytorch'),
        rpn_head=dict(bbox_coder=dict(clip_border=True)),
        roi_head=dict(
            bbox_head=dict(bbox_coder=dict(clip_border=True), num_classes=1))))
data_root = 'data/MOT20/'
data = dict(
    train=dict(
        ann_file=data_root + 'annotations/half-train_cocoformat.json',
        img_prefix=data_root + 'train'),
    val=dict(
        ann_file=data_root + 'annotations/half-val_cocoformat.json',
        img_prefix=data_root + 'train'),
    test=dict(
        ann_file=data_root + 'annotations/half-val_cocoformat.json',
        img_prefix=data_root + 'train'))
load_from = (
    'http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/'
    'faster_rcnn_x101_32x4d_fpn_2x_coco/faster_rcnn_x101_32x4d_fpn_2x_coco_'
    'bbox_mAP-0.412_20200506_041400-64a12c0b.pth')
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=1.0 / 100,
    step=[6])
# runtime settings
total_epochs = 8
