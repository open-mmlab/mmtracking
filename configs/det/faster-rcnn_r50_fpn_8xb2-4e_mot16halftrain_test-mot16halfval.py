_base_ = [
    './faster-rcnn_resnet50-fpn_8xb2-4e_mot17halftrain_test-mot17halfval.py'
]
# data
data_root = 'data/MOT16/'
train_dataloader = dict(dataset=dict(data_root=data_root))
val_dataloader = dict(dataset=dict(data_root=data_root))
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root +
                     'annotations/half-val_cocoformat.json')
test_evaluator = val_evaluator
