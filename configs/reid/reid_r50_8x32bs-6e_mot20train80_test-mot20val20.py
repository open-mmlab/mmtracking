_base_ = ['./resnet50_b32x8_MOT17.py']
model = dict(head=dict(num_classes=1701))
# data
data_root = 'data/MOT20/'
train_dataloader = dict(dataset=dict(data_root=data_root))
val_dataloader = dict(dataset=dict(data_root=data_root))
test_dataloader = val_dataloader

# train, val, test setting
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=6, val_interval=7)
