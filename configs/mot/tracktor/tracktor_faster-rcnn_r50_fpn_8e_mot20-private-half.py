_base_ = ['./tracktor_faster-rcnn_r50_fpn_4e_mot17-private-half.py']
model = dict(
    pretrains=dict(
        detector=  # noqa: E251
        # 'https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot15-half-f48f6578.pth',  # noqa: E501
        'work_dirs/change_configs/faster-rcnn_r50_fpn_8e_mot20-half/latest.pth',  # noqa: E501
        reid=  # noqa: E251
        # 'https://download.openmmlab.com/mmtracking/mot/reid/reid_r50_6e_mot15-30ba14d3.pth'  # noqa: E501
        'work_dirs/change_configs/resnet50_b32x8_MOT20_1/latest.pth'
    ),  # noqa: E501
    detector=dict(
        rpn_head=dict(bbox_coder=dict(clip_border=True)),
        roi_head=dict(
            bbox_head=dict(bbox_coder=dict(clip_border=True), num_classes=1))),
    reid=dict(head=dict(num_classes=1705)))
# data
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
# learning policy
lr_config = dict(step=[6])
# runtime settings
total_epochs = 8
