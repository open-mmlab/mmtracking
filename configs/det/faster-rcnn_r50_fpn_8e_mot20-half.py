USE_MMDET = True
_base_ = ['./faster-rcnn_r50_fpn_4e_mot17-half.py']
model = dict(
    detector=dict(
        rpn_head=dict(bbox_coder=dict(clip_border=True)),
        roi_head=dict(
            bbox_head=dict(bbox_coder=dict(clip_border=True), num_classes=1))))
# data
data_root = 'data/MOT20/'
data = dict(
    train=dict(
        ann_file=data_root + 'annotations/half-train_cocoformat.json',
        img_prefix=data_root + 'train'),
    val=dict(
        ann_file=data_root + 'annotations/half-val_cocoformat.json',
        img_prefix=data_root + 'train'),
    test=dict(
        ann_file=data_root + 'annotations/half-val_cocoformat.json',
        img_prefix=data_root + 'train'))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=1.0 / 100,
    step=[6])
# runtime settings
total_epochs = 8
