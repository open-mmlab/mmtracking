_base_ = ['./faster-rcnn_r50_fpn_4e_mot17-half.py']
# data
data_root = 'data/MOT15/'
train_dataloader = dict(dataset=dict(data_root=data_root))
val_dataloader = dict(dataset=dict(data_root=data_root))
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root +
                     'annotations/half-val_cocoformat.json')
test_evaluator = val_evaluator
