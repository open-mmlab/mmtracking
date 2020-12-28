_base_ = ['./tracktor_faster-rcnn_r50_fpn_4e_mot17-private-half.py']
data_root = 'data/MOT17/'
data = dict(
    train=dict(ann_file=data_root + 'annotations/train_cocoformat.json'),
    val=dict(ann_file=data_root + 'annotations/train_cocoformat.json'),
    test=dict(
        ann_file=data_root + 'annotations/test_cocoformat.json',
        img_prefix=data_root + 'test'))
