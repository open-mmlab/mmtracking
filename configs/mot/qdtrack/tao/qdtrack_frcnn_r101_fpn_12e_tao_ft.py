# model settings
_base_ = './qdtrack_frcnn_r101_fpn_24e_lvis.py'
model = dict(freeze_detector=True)
# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadMultiImagesFromFile'),
    dict(type='SeqLoadAnnotations', with_bbox=True, with_track=True),
    dict(
        type='SeqResize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                   (1333, 768), (1333, 800)],
        share_params=True,
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
    dict(type='SeqNormalize', **img_norm_cfg),
    dict(type='SeqPad', size_divisor=32),
    dict(type='MatchInstances', skip_nomatch=True),
    dict(
        type='VideoCollect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_match_indices'],
    ),
    dict(type='SeqDefaultFormatBundle', ref_prefix='ref'),
]
dataset_type = 'TaoDataset'
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        _delete_=True,
        type='ClassBalancedDataset',
        oversample_thr=1e-3,
        dataset=dict(
            type=dataset_type,
            classes='data/tao/annotations/tao_classes.txt',
            ann_file='data/tao/annotations/qdtrack_482/train_482_ours.json',
            img_prefix='data/tao/frames/',
            key_img_sampler=dict(interval=1),
            ref_img_sampler=dict(
                num_ref_imgs=1, frame_range=[-1, 1], method='uniform'),
            pipeline=train_pipeline)))
optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 1000,
    step=[8, 11])
total_epochs = 12
load_from = None
evaluation = dict(metric=['track'], start=1, interval=1)
work_dir = './work_dirs/tao/finetune'
