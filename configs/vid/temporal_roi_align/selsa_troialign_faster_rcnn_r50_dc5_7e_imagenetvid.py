_base_ = [
    '../../_base_/models/faster_rcnn_r50_dc5.py',
    '../../_base_/datasets/imagenet_vid_fgfa_style.py',
    '../../_base_/default_runtime.py'
]
model = dict(
    type='SELSA',
    detector=dict(
        roi_head=dict(
            type='SelsaRoIHead',
            bbox_roi_extractor=dict(
                type='TemporalRoIAlign',
                num_most_similar_points=2,
                num_temporal_attention_blocks=4,
                roi_layer=dict(
                    type='RoIAlign', output_size=7, sampling_ratio=2),
                out_channels=512,
                featmap_strides=[16]),
            bbox_head=dict(
                type='SelsaBBoxHead',
                num_shared_fcs=3,
                aggregator=dict(
                    type='SelsaAggregator',
                    in_channels=1024,
                    num_attention_blocks=16)))))

# dataset settings
data = dict(
    val=dict(
        ref_img_sampler=dict(
            _delete_=True,
            num_ref_imgs=14,
            frame_range=[-7, 7],
            method='test_with_adaptive_stride')),
    test=dict(
        ref_img_sampler=dict(
            _delete_=True,
            num_ref_imgs=14,
            frame_range=[-7, 7],
            method='test_with_adaptive_stride')))

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
