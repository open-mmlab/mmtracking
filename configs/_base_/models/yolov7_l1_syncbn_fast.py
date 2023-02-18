# parameters that often need to be modified
img_scale = (1088, 1088)  # width, height

# different from yolov5
anchors = [
    [(19, 27), (44, 40), (38, 94)],  # P3/8
    [(96, 68), (86, 152), (180, 137)],  # P4/16
    [(140, 301), (303, 264), (238, 542)],  # P5/32
]
strides = [8, 16, 32]
num_det_layers = 3
num_classes = 1

model = dict(
    data_preprocessor=dict(
        type='TrackDataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    detector=dict(
        type='YOLODetector',
        _scope_='mmyolo',
        backbone=dict(
            type='YOLOv7Backbone',
            arch='L',
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=dict(type='SiLU', inplace=True)),
        neck=dict(
            type='YOLOv7PAFPN',
            block_cfg=dict(
                type='ELANBlock',
                middle_ratio=0.5,
                block_ratio=0.25,
                num_blocks=4,
                num_convs_in_block=1),
            upsample_feats_cat_first=False,
            in_channels=[512, 1024, 1024],
            # The real output channel will be multiplied by 2
            out_channels=[128, 256, 512],
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=dict(type='SiLU', inplace=True)),
        bbox_head=dict(
            type='YOLOv7Head',
            head_module=dict(
                type='YOLOv7HeadModule',
                num_classes=num_classes,
                in_channels=[256, 512, 1024],
                featmap_strides=strides,
                num_base_priors=3),
            prior_generator=dict(
                type='mmdet.YOLOAnchorGenerator',
                base_sizes=anchors,
                strides=strides),
            obj_level_weights=[4., 1., 0.4],
            # BatchYOLOv7Assigner params
            prior_match_thr=4.,
            simota_candidate_topk=10,
            simota_iou_weight=3.0,
            simota_cls_weight=1.0),
        test_cfg=dict(
            multi_label=False,
            nms_pre=30000,
            score_thr=0.01,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=300)))