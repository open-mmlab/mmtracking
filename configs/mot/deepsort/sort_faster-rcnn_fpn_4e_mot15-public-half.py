_base_ = ['./sort_faster-rcnn_fpn_4e_mot17-public-half.py']
model = dict(
    pretrains=dict(
        detector=  # noqa: E251
        'work_dirs/detector/faster-rcnn_r50_fpn_4e_mot15-half/latest.pth'  # noqa: E501
    ))
data_root = 'data/MOT15/'
data = dict(
    val=dict(
        detection_file=data_root + 'annotations/half-val_detections.pkl'),
    test=dict(
        detection_file=data_root + 'annotations/half-val_detections.pkl'))
