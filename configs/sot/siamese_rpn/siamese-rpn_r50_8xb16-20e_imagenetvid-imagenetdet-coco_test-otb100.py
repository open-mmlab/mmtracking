_base_ = [
    '../../_base_/datasets/otb100.py',
    './siamese-rpn_r50_8xb28-20e_imagenetvid-imagenetdet-coco_base.py'
]

crop_size = 511
exemplar_size = 127
search_size = 255

# model settings
model = dict(
    test_cfg=dict(rpn=dict(penalty_k=0.4, window_influence=0.5, lr=0.4)))

data_root = {{_base_.data_root}}
train_pipeline = [
    dict(
        type='PairSampling',
        frame_range=100,
        pos_prob=0.8,
        filter_template_img=False),
    dict(
        type='TransformBroadcaster',
        share_random_params=False,
        transforms=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadTrackAnnotations', with_instance_id=False),
            dict(
                type='CropLikeSiamFC',
                context_amount=0.5,
                exemplar_size=exemplar_size,
                crop_size=crop_size)
        ]),
    dict(
        type='TransformBroadcaster',
        share_random_params=True,
        transforms=dict(type='GrayAug', prob=0.2)),
    dict(
        type='SeqShiftScaleAug',
        target_size=[exemplar_size, search_size],
        shift=[4, 64],
        scale=[0.05, 0.18]),
    dict(type='SeqColorAug', prob=[1.0, 1.0]),
    dict(type='SeqBlurAug', prob=[0.0, 0.2]),
    dict(type='PackTrackInputs', ref_prefix='search', num_template_frames=1)
]

# dataloader
train_dataloader = dict(
    batch_size=16,
    dataset=dict(datasets=[
        dict(
            type='SOTImageNetVIDDataset',
            data_root=data_root,
            ann_file='ILSVRC/annotations/imagenet_vid_train.json',
            data_prefix=dict(img_path='ILSVRC/Data/VID'),
            pipeline=train_pipeline,
            test_mode=False),
        dict(
            type='SOTCocoDataset',
            data_root=data_root,
            ann_file='coco/annotations/instances_train2017.json',
            data_prefix=dict(img_path='coco/train2017'),
            pipeline=train_pipeline,
            test_mode=False),
        dict(
            type='SOTCocoDataset',
            data_root=data_root,
            ann_file='ILSVRC/annotations/imagenet_det_30plus1cls.json',
            data_prefix=dict(img_path='ILSVRC/Data/DET'),
            pipeline=train_pipeline,
            test_mode=False)
    ]))
