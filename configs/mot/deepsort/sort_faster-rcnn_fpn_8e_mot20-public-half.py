_base_ = ['./sort_faster-rcnn_fpn_4e_mot17-public-half.py']
model = dict(
    pretrains=dict(
        detector=  # noqa: E251
        'work_dirs/detector/faster-rcnn_r50_fpn_8e_mot20-half_clip_border_interval8/faster-rcnn_r50_fpn_8e_mot20-half_clip_border_interval8.pth'  # noqa: E501
    ),
    detector=dict(
        rpn_head=dict(bbox_coder=dict(clip_border=True)),
        roi_head=dict(
            bbox_head=dict(bbox_coder=dict(
                clip_border=True), num_classes=1))))
data_root = 'data/MOT20/'
data = dict(
    train=dict(
        ann_file=data_root + 'annotations/half-train_cocoformat.json',
        img_prefix=data_root + 'train'),
    val=dict(
        ann_file=data_root + 'annotations/half-val_cocoformat.json',
        img_prefix=data_root + 'train',
        detection_file=data_root + 'annotations/half-val_detections.pkl'),
    test=dict(
        ann_file=data_root + 'annotations/half-val_cocoformat.json',
        img_prefix=data_root + 'train',
        detection_file=data_root + 'annotations/half-val_detections.pkl'))
