_base_ = ['./tracktor_faster-rcnn_r50_fpn_4e_mot20-private-half.py']
model = dict(
    pretrains=dict(
        detector=  # noqa: E251
        'https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_8e_mot20-ef875499.pth'  # noqa: E501
    ))
data_root = 'data/MOT20/'
test_set = 'train'
data = dict(
    train=dict(ann_file=data_root + 'annotations/train_cocoformat.json'),
    val=dict(ann_file=data_root + 'annotations/train_cocoformat.json'),
    test=dict(
        ann_file=data_root + f'annotations/{test_set}_cocoformat.json',
        img_prefix=data_root + test_set))
