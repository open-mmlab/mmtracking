_base_ = ['./resnet50_b32*8_MOT17.py']

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
            num_classes=378,
            losses=[dict(type='TripletLoss', margin=0.5, loss_weight=1.0),
                    dict(type='CrossEntropyLoss', loss_weight=1.0)],
            cal_acc=True,
            norm_cfg=dict(type='BN1d'),
            act_cfg=dict(type='ReLU')))
)