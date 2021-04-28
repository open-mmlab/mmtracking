_base_ = ['./sort_faster-rcnn_fpn_4e_mot17-public-half.py']
model = dict(
    pretrains=dict(
        detector=  # noqa: E251
        'work_dirs/detector/faster-rcnn_r50_fpn_4e_mot20/latest.pth'  # noqa: E501
    ))
data_root = 'data/MOT20/'
test_set = 'test'
data = dict(
    train=dict(ann_file=data_root + 'annotations/train_cocoformat.json'),
    val=dict(
        ann_file=data_root + 'annotations/train_cocoformat.json',
        img_prefix=data_root + test_set,
        detection_file=data_root + 'annotations/train_detections.pkl'),
    test=dict(
        ann_file=data_root + f'annotations/{test_set}_cocoformat.json',
        img_prefix=data_root + test_set,
        detection_file=data_root + f'annotations/{test_set}_detections.pkl'))
