_base_ = [
    '../../_base_/models/mask_rcnn_r50_fpn.py',
    '../../_base_/datasets/youtube_vis.py', '../../_base_/default_runtime.py'
]
model = dict(
    type='MaskTrackRCNN',
    detector=dict(
        roi_head=dict(
            bbox_head=dict(num_classes=40), mask_head=dict(num_classes=40)),
        train_cfg=dict(
            rpn=dict(sampler=dict(num=64)),
            rpn_proposal=dict(nms_pre=200, max_per_img=200),
            rcnn=dict(sampler=dict(num=128))),
        test_cfg=dict(
            rpn=dict(nms_pre=200, max_per_img=200), rcnn=dict(score_thr=0.01)),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'  # noqa: E501
        )),
    track_head=dict(
        type='RoITrackHead',
        roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        embed_head=dict(
            type='RoIEmbedHead',
            num_fcs=2,
            roi_feat_size=7,
            in_channels=256,
            fc_out_channels=1024),
        train_cfg=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=128,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    tracker=dict(
        type='MaskTrackRCNNTracker',
        match_weights=dict(det_score=1.0, iou=2.0, det_label=10.0),
        num_frames_retain=20))

# optimizer
optimizer = dict(type='SGD', lr=0.00125, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
# runtime settings
total_epochs = 12
evaluation = dict(metric=['track_segm'], interval=13)
