_base_ = ['./qdtrack_faster-rcnn_r50_fpn_4e_mot17-private-half.py']
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadMultiImagesFromFile', to_float32=True),
    dict(type='SeqLoadAnnotations', with_bbox=True, with_track=True),
    dict(
        type='SeqResize',
        img_scale=(1088, 1088),
        share_params=True,
        ratio_range=(0.8, 1.2),
        keep_ratio=True,
        bbox_clip_border=False),
    dict(type='SeqPhotoMetricDistortion', share_params=True),
    dict(
        type='SeqRandomCrop',
        share_params=False,
        crop_size=(1088, 1088),
        bbox_clip_border=False),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
    dict(type='SeqNormalize', **img_norm_cfg),
    dict(type='SeqPad', size_divisor=32),
    dict(type='MatchInstances', skip_nomatch=True),
    dict(
        type='VideoCollect',
        keys=[
            'img', 'gt_bboxes', 'gt_labels', 'gt_match_indices',
            'gt_instance_ids'
        ]),
    dict(type='SeqDefaultFormatBundle', ref_prefix='ref')
]
mot_cfg = dict(
    type='MOTChallengeDataset',
    classes=('pedestrian', ),
    visibility_thr=-1,
    ann_file='data/MOT17/annotations/half-train_cocoformat.json',
    img_prefix='data/MOT17/train',
    ref_img_sampler=dict(num_ref_imgs=1, frame_range=10, method='uniform'),
    pipeline=train_pipeline)
crowdhuman_cfg = dict(
    type='CocoVideoDataset',
    load_as_video=False,
    classes=('pedestrian', ),
    ann_file='data/crowdhuman/annotations/crowdhuman_train.json',
    img_prefix='data/crowdhuman/train',
    pipeline=train_pipeline)
data = dict(
    train=dict(
        _delete_=True,
        type='ConcatDataset',
        datasets=[mot_cfg, crowdhuman_cfg],
        saparate_eval=False))
