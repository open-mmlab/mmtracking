USE_MMCLS = True
_base_ = [
    '../_base_/datasets/mot_challenge_reid.py', '../_base_/default_runtime.py'
]
model = dict(
    reid=dict(
        type='BaseReID',
        backbone=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(3, ),
            style='pytorch'),
        neck=dict(type='GlobalAveragePooling', kernel_size=(8, 4), stride=1),
        head=dict(
            type='LinearReIDHead',
            num_fcs=1,
            in_channels=2048,
            fc_channels=1024,
            out_channels=128,
            num_classes=68,
            loss_cls=dict(type='CrossEntropyLoss', loss_weight=1.0),
            loss_triplet=dict(type='TripletLoss', margin=1.0, loss_weight=1.0),
            cal_acc=True,
            norm_cfg=dict(type='BN1d'),
            act_cfg=dict(type='ReLU'))))
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        data_prefix='data/MOT17/reid/img',
        ann_file='data/MOT17/reid/meta/debug_train.txt'),
    val=dict(
        data_prefix='data/MOT17/reid/img',
        ann_file='data/MOT17/reid/meta/debug_val.txt'),
    test=dict(
        data_prefix='data/MOT17/reid/img',
        ann_file='data/MOT17/reid/meta/debug_val.txt'))
# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[3, 6, 9])
total_epochs = 10
evaluation = dict(interval=1, metric=['mAP', 'CMC'])

load_from = 'https://download.openmmlab.com/mmclassification/v0/resnet/' \
            'resnet50_batch256_imagenet_20200708-cfb998bf.pth'
