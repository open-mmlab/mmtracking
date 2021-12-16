# model settings
img_scale = (640, 640)

model = dict(
    detector=dict(
        type='YOLOX',
        input_size=img_scale,
        random_size_range=(15, 25),
        random_size_interval=10,
        backbone=dict(
            type='CSPDarknet', deepen_factor=1.33, widen_factor=1.25),
        neck=dict(
            type='YOLOXPAFPN',
            in_channels=[320, 640, 1280],
            out_channels=320,
            num_csp_blocks=4),
        bbox_head=dict(
            type='YOLOXHead',
            num_classes=80,
            in_channels=320,
            feat_channels=320),
        train_cfg=dict(
            assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
        test_cfg=dict(
            score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65))))
