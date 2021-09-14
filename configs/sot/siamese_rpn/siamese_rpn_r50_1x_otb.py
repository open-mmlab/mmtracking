_base_ = ['./siamese_rpn_r50_1x_lasot.py']

crop_size = 511
exemplar_size = 127
search_size = 255

# model settings
model = dict(
    head=dict(weighted_sum=False),
    test_cfg=dict(rpn=dict(penalty_k=0.24, window_influence=0.5, lr=0.25)))

data_root = 'data/'
train_pipeline = [
    dict(type='LoadMultiImagesFromFile', to_float32=True),
    dict(type='SeqLoadAnnotations', with_bbox=True),
    dict(
        type='SeqCropLikeSiamFC',
        context_amount=0.5,
        exemplar_size=exemplar_size,
        crop_size=crop_size),
    dict(type='SeqGrayAug', prob=0.2),
    dict(
        type='SeqShiftScaleAug',
        target_size=[exemplar_size, search_size],
        shift=[4, 64],
        scale=[0.05, 0.18]),
    dict(type='SeqColorAug', prob=[1.0, 1.0]),
    dict(type='SeqBlurAug', prob=[0.0, 0.2]),
    dict(type='VideoCollect', keys=['img', 'gt_bboxes', 'is_positive_pairs']),
    dict(type='ConcatVideoReferences'),
    dict(type='SeqDefaultFormatBundle', ref_prefix='search')
]
# dataset settings
data = dict(
    samples_per_gpu=16,
    val=dict(
        type='OTB100Dataset',
        ann_file=data_root + 'otb100/annotations/otb100.json',
        img_prefix=data_root + 'otb100/data'),
    test=dict(
        type='OTB100Dataset',
        ann_file=data_root + 'otb100/annotations/otb100.json',
        img_prefix=data_root + 'otb100/data'))
# optimizer
optimizer_config = dict(
    backbone_train_layers=['layer1', 'layer2', 'layer3', 'layer4'])
# learning policy
lr_config = dict(lr_configs=[
    dict(type='step', start_lr_factor=0.2, end_lr_factor=1.0, end_epoch=5),
    dict(type='log', start_lr_factor=1.0, end_lr_factor=0.5, end_epoch=20)
])
