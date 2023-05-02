USE_MMDET = True
_base_ = ['./faster-rcnn_r50_fpn_4e_mot17-half.py']

# data
data_root = 'data/AnimalTrack/'
data = dict(
    train=dict(
        ann_file=data_root + 'annotations/half-train_cocoformat.json',
        img_prefix=data_root + 'train',
        classes=('chicken', 'deer', 'dolphin', 'duck', 'goose', 'horse', 'penguin', 'pig', 'rabbit', 'zebra'),
    ),
    val=dict(
        ann_file=data_root + 'annotations/half-val_cocoformat.json',
        img_prefix=data_root + 'train',
        classes=('chicken', 'deer', 'dolphin', 'duck', 'goose', 'horse', 'penguin', 'pig', 'rabbit', 'zebra'),
    ),
    test=dict(
        ann_file=data_root + 'annotations/test_cocoformat.json',
        img_prefix=data_root + 'test',
        classes=('chicken', 'deer', 'dolphin', 'duck', 'goose', 'horse', 'penguin', 'pig', 'rabbit', 'zebra'),
    )
)

# train
total_epochs = 50
checkpoint_config = dict(interval=10)
max_keep_ckpts = 5
optimizer = dict(type='SGD', lr=1e-3, weight_decay=0.0001)