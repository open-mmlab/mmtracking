_base_ = ['./tracktor_faster-rcnn_r50_fpn_4e_mot17-private-half.py']

model = dict(
    detector=dict(
        rpn_head=dict(bbox_coder=dict(clip_border=True)),
        roi_head=dict(
            bbox_head=dict(bbox_coder=dict(clip_border=True), num_classes=1)),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_8e_mot20_20210804_162232-7fde5e8d.pth'  # noqa: E501
        )),
    reid=dict(
        head=dict(num_classes=1705),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://download.openmmlab.com/mmtracking/mot/reid/reid_r50_6e_mot20_20210803_212426-c83b1c01.pth'  # noqa: E501
        )))
data_root = 'data/MOT20/'
test_set = 'test'
data = dict(
    train=dict(ann_file=data_root + 'annotations/train_cocoformat.json'),
    val=dict(
        ann_file=data_root + 'annotations/train_cocoformat.json',
        detection_file=data_root + 'annotations/train_detections.pkl'),
    test=dict(
        ann_file=data_root + f'annotations/{test_set}_cocoformat.json',
        img_prefix=data_root + test_set,
        detection_file=data_root + f'annotations/{test_set}_detections.pkl'))
# learning policy
lr_config = dict(step=[6])
# runtime settings
total_epochs = 8
