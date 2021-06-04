USE_MMCLS = True
_base_ = ['../_base_/datasets/mot_challenge_reid.py', '../_base_/default_runtime.py']
model = dict(
    # pretrains=dict(
    #     reid=
    # ),
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
            num_classes=436,
            norm_cfg=dict(type='BN1d'),
            act_cfg=dict(type='ReLU')))
)
# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 1000,
    step=[30, 60, 90])
total_epochs = 100

load_from = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_batch256_imagenet_20200708-cfb998bf.pth'
