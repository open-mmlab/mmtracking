_base_ = [
    './strongsort_yolox_x_8xb4-80e_crowdhuman-mot17halftrain'
    '_test-mot17halfval.py'
]

model = dict(
    data_preprocessor=dict(
        type='TrackDataPreprocessor',
        pad_size_divisor=32,
        batch_augments=[
            dict(
                type='mmdet.BatchSyncRandomResize',
                random_size_range=(640, 1152))
        ]),
    detector=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://download.openmmlab.com/mmtracking/mot/strongsort/mot_dataset/yolox_x_crowdhuman_mot20-private_20220812_192123-77c014de.pth'  # noqa: E501
        )),
    reid=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://download.openmmlab.com/mmtracking/mot/reid/reid_r50_6e_mot20_20210803_212426-c83b1c01.pth'  # noqa: E501
        )),
)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='mmdet.Resize', scale=(896, 1600), keep_ratio=True),
    dict(
        type='mmdet.Pad',
        size_divisor=32,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='PackTrackInputs', pack_single_img=True)
]

val_dataloader = dict(
    dataset=dict(
        data_root='data/MOT17',
        ann_file='annotations/train_cocoformat.json',
        data_prefix=dict(img_path='train'),
        pipeline=test_pipeline))
test_dataloader = dict(
    dataset=dict(
        data_root='data/MOT20',
        ann_file='annotations/test_cocoformat.json',
        data_prefix=dict(img_path='test'),
        pipeline=test_pipeline))

test_evaluator = dict(format_only=True, outfile_prefix='./mot_20_test_res')
