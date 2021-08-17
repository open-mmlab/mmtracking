_base_ = [
    '../../_base_/models/faster_rcnn_r50_dc5.py',
    '../../_base_/datasets/imagenet_vid_dff_style.py',
    '../../_base_/default_runtime.py'
]
model = dict(
    type='DFF',
    detector=dict(
        train_cfg=dict(
            rpn_proposal=dict(max_per_img=1000),
            rcnn=dict(sampler=dict(num=512)))),
    motion=dict(
        type='FlowNetSimple',
        img_scale_factor=0.5,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://download.openmmlab.com/mmtracking/pretrained_weights/flownet_simple.pth'  # noqa: E501
        )),
    train_cfg=None,
    test_cfg=dict(key_frame_interval=10))

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[2, 5])
# runtime settings
total_epochs = 7
evaluation = dict(metric=['bbox'], interval=7)
