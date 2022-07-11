_base_ = ['./resnet50_b32x8_MOT17.py']
model = dict(head=dict(num_classes=368))
# data
data_root = 'data/MOT15/'
train_dataloader = dict(dataset=dict(data_root=data_root))
val_dataloader = dict(dataset=dict(data_root=data_root))
test_dataloader = val_dataloader
