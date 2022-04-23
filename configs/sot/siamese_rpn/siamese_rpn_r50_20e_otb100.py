_base_ = ['./siamese_rpn_r50_20e_lasot.py']

crop_size = 511
exemplar_size = 127
search_size = 255

# model settings
model = dict(
    test_cfg=dict(rpn=dict(penalty_k=0.4, window_influence=0.5, lr=0.4)))

data_root = 'data/'
train_pipeline = [
    dict(
        type='PairSampling',
        frame_range=100,
        pos_prob=0.8,
        filter_template_img=False),
    dict(type='LoadMultiImagesFromFile', to_float32=True),
    dict(type='SeqLoadAnnotations', with_bbox=True, with_label=False),
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
    dict(type='ConcatSameTypeFrames'),
    dict(type='SeqDefaultFormatBundle', ref_prefix='search')
]
# dataset settings
data = dict(
    samples_per_gpu=16,
    train=dict(dataset_cfgs=[
        dict(
            type='SOTImageNetVIDDataset',
            ann_file=data_root + 'ILSVRC/annotations/imagenet_vid_train.json',
            img_prefix=data_root + 'ILSVRC/Data/VID',
            pipeline=train_pipeline,
            split='train',
            test_mode=False),
        dict(
            type='SOTCocoDataset',
            ann_file=data_root + 'coco/annotations/instances_train2017.json',
            img_prefix=data_root + 'coco/train2017',
            pipeline=train_pipeline,
            split='train',
            test_mode=False),
        dict(
            type='SOTCocoDataset',
            ann_file=data_root +
            'ILSVRC/annotations/imagenet_det_30plus1cls.json',
            img_prefix=data_root + 'ILSVRC/Data/DET',
            pipeline=train_pipeline,
            split='train',
            test_mode=False)
    ]),
    val=dict(
        type='OTB100Dataset',
        ann_file=data_root + 'otb100/annotations/otb100_infos.txt',
        img_prefix=data_root + 'otb100',
        only_eval_visible=False),
    test=dict(
        type='OTB100Dataset',
        ann_file=data_root + 'otb100/annotations/otb100_infos.txt',
        img_prefix=data_root + 'otb100',
        only_eval_visible=False))
