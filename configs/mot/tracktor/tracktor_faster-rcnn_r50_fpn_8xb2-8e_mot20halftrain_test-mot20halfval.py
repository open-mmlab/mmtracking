_base_ = [
    './tracktor_faster-rcnn_r50-fpn_8xb2-4e_mot17halftrain'
    '_test-mot17halfval.py'
]

model = dict(
    detector=dict(
        rpn_head=dict(bbox_coder=dict(clip_border=True)),
        roi_head=dict(
            bbox_head=dict(bbox_coder=dict(clip_border=True), num_classes=1)),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_8e_mot20-half_20210805_001244-2c323fd1.pth'  # noqa: E501
        )),
    reid=dict(
        head=dict(num_classes=1705),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://download.openmmlab.com/mmtracking/mot/reid/reid_r50_6e_mot20_20210803_212426-c83b1c01.pth'  # noqa: E501
        )))

# dataloader
data_root = 'data/MOT20/'
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/half-val_cocoformat.json',
    ))
test_dataloader = val_dataloader
