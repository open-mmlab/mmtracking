USE_MMDET = True
_base_ = ['./faster-rcnn_r50_fpn_4e_mot17-half.py']
model = dict(
    detector=dict(
        pretrained='torchvision://resnet101',
        backbone=dict(depth=101),
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
load_from = ('http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/'
             'faster_rcnn_r101_fpn_2x_coco/faster_rcnn_r101_fpn_2x_coco_bbox_'
             'mAP-0.398_20200504_210455-1d2dac9c.pth')
