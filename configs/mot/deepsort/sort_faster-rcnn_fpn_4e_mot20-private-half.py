_base_ = ['./sort_faster-rcnn_fpn_4e_mot17-private-half.py']
model = dict(
    pretrains=dict(
        detector=  # noqa: E251
        'work_dirs/detector/faster-rcnn_r50_fpn_4e_mot15-half/latest.pth'  # noqa: E501
    ))
data_root = 'data/MOT17/'
data = dict(
    val=dict(
        ann_file=data_root + 'annotations/half-val_cocoformat.json',
        img_prefix=data_root + 'train',
        detection_file=data_root + 'annotations/half-val_detections.pkl'),
    test=dict(
        ann_file=data_root + 'annotations/half-val_cocoformat.json',
        img_prefix=data_root + 'train',
        detection_file=data_root + 'annotations/half-val_detections.pkl'))