_base_ = ['./sort_faster-rcnn_fpn_4e_mot17-public-half']
data_root = 'data/MOT15/'
data = dict(
    val=dict(
        detection_file=data_root + 'annotations/half-val_detections.pkl'),
    test=dict(
        detection_file=data_root + 'annotations/half-val_detections.pkl'))
