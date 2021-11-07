from mmtrack.models.track_heads.stark_head import CornerPredictorHead


cudnn_benchmark = True
crop_size = 511
exemplar_size = 127
search_size = 255

# model setting
model = dict(
    type='Stark',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=3,
        strides=(1, 2, 2),
        dilations=[1, 1, 1],
        out_indices=[2],
        norm_eval=True,
        norm_cfg=dict(type='BN', requires_grad=False),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='ChannelMapper',
        in_channels=[1024],
        out_channels=256,
        kernel_size=1,
        act_cfg=None),
    head=dict(
        type='StarkHead',
        num_querys=1,
        transformer=dict(
            type='StarkTransformer',
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            attn_drop=0.1,
                            dropout_layer=dict(type='Dropout', drop_prob=0.1))
                    ],
                    ffn_cfgs=dict(feedforward_channels=2048, embed_dims=256, ffn_drop=0.1),
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DetrTransformerDecoder',
                return_intermediate=False,
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiheadAttention',
                        embed_dims=256,
                        num_heads=8,
                        attn_drop=0.1,
                        dropout_layer=dict(type='Dropout', drop_prob=0.1)),
                    ffn_cfgs=dict(feedforward_channels=2048, embed_dims=256, ffn_drop=0.1),
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
        ),
        positional_encoding=dict(
            type='SinePositionalEncoding', num_feats=128, normalize=True),
        bbox_head=dict(type='CornerPredictorHead',
                inplanes=256,
                channel=256,
                feat_size=20,
                stride=16),
        cls_head=dict(type='ScoreHead', 
                input_dim=256,
                hidden_dim=256,
                output_dim=1,
                num_layers=3,
                BN=False)),
        test_cfg=dict(
            epoch=50,
            search_factor=5.0,
            search_size=320,
            template_factor=2.0,
            template_size=128,
            update_intervals=[200]))

train_pipeline = [
    dict(type='LoadMultiImagesFromFile', to_float32=True),
    dict(type='SeqLoadAnnotations', with_bbox=True),
    dict(type='SeqGrayAug', prob=0.05),
    dict(type='SeqRandomFlip', prob=0.5),

    dict(
        type='SeqCropLikeSiamFC',
        context_amount=0.5,
        exemplar_size=exemplar_size,
        crop_size=crop_size),
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

img_norm_cfg = dict(mean=[0, 0, 0], std=[1, 1, 1], to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='MultiScaleFlipAug',
        scale_factor=1,
        flip=False,
        transforms=[
            dict(type='Normalize', **img_norm_cfg),
            dict(type='VideoCollect', keys=['img', 'gt_bboxes']),
            dict(type='ImageToTensor', keys=['img'])
        ])
]

data_root = 'data/'
# dataset settings
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=[
        dict(
            type='RepeatDataset',
            times=39,
            dataset=dict(
                type='SOTTrainDataset',
                ann_file=data_root +
                'ILSVRC/annotations/imagenet_vid_train.json',
                img_prefix=data_root + 'ILSVRC/Data/VID',
                pipeline=train_pipeline,
                ref_img_sampler=dict(
                    frame_range=100,
                    pos_prob=0.8,
                    filter_key_img=False,
                    return_key_img=True),
            )),
        dict(
            type='SOTTrainDataset',
            ann_file=data_root + 'coco/annotations/instances_train2017.json',
            img_prefix=data_root + 'coco/train2017',
            pipeline=train_pipeline,
            ref_img_sampler=dict(
                frame_range=0,
                pos_prob=0.8,
                filter_key_img=False,
                return_key_img=True),
        ),
        dict(
            type='SOTTrainDataset',
            ann_file=data_root +
            'ILSVRC/annotations/imagenet_det_30plus1cls.json',
            img_prefix=data_root + 'ILSVRC/Data/DET',
            pipeline=train_pipeline,
            ref_img_sampler=dict(
                frame_range=0,
                pos_prob=0.8,
                filter_key_img=False,
                return_key_img=True),
        ),
    ],
    test=dict(
        type='GOT10kDataset',
        test_load_ann=True,
        ann_file=data_root + 'got10k/annotations/got10k_test.json',
        img_prefix=data_root + 'got10k/test',
        pipeline=test_pipeline,
        ref_img_sampler=None,
        test_mode=True))
