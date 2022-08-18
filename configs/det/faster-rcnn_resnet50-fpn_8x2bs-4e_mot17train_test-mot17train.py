_base_ = ['./faster-rcnn_r50_fpn_4e_mot17-half.py']
# data
data_root = 'data/MOT17/'
train_dataloader = dict(
    dataset=dict(ann_file='annotations/train_cocoformat.json'))
val_dataloader = dict(
    dataset=dict(ann_file='annotations/train_cocoformat.json'))
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'annotations/train_cocoformat.json')
test_evaluator = val_evaluator
