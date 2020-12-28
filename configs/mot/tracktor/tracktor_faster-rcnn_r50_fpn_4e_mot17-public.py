_base_ = ['./tracktor_faster-rcnn_r50_fpn_4e_mot17-public-half.py']
model = dict(
    pretrains=dict(
        detector=  # noqa: E251
        'https://download.openmmlab.com/mmtracking/v0.5/faster-rcnn_r50_fpn_4e_mot17-ffa52ae7.pth'  # noqa: E501
    ))
data_root = 'data/MOT17/'
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
