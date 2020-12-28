USE_MMDET = True
_base_ = ['./faster-rcnn_r50_fpn_4e_mot17-half.py']
model = dict(
    detector=dict(
        pretrained='torchvision://resnet101', backbone=dict(depth=101)))
load_from = ('http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/'
             'faster_rcnn_r101_fpn_2x_coco/faster_rcnn_r101_fpn_2x_coco_bbox_'
             'mAP-0.398_20200504_210455-1d2dac9c.pth')
