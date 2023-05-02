import torch

TRAIN_REID = True
_base_ = ['./resnet50_b32x8_MOT17.py']
model = dict(reid=dict(head=dict(num_classes=816)))

# data
data_root = 'data/AnimalTrack/'
data = dict(
    train=dict(
        data_prefix=data_root + 'reid/imgs',
        ann_file=data_root + 'reid/meta/train_80.txt'),
    val=dict(
        data_prefix=data_root + 'reid/imgs',
        ann_file=data_root + 'reid/meta/val_20.txt'),
    test=dict(
        data_prefix=data_root + 'reid/imgs',
        ann_file=data_root + 'reid/meta/val_20.txt'))

checkpoint_config = dict(interval=1)
seed = 0
gpu_ids = range(1)
device = "cuda" if torch.cuda.is_available() else "cpu"

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=1.0 / 200,
    step=[1])
total_epochs = 10