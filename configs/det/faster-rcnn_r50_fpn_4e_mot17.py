USE_MMDET = True
_base_ = ['./faster-rcnn_r50_fpn_4e_mot17-half.py']
# data
data_root = 'data/MOT17/'
data = dict(
    train=dict(ann_file=data_root + 'annotations/train_cocoformat.json'),
    val=dict(ann_file=data_root + 'annotations/train_cocoformat.json'),
    test=dict(ann_file=data_root + 'annotations/train_cocoformat.json'))
