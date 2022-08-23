_base_ = [
    './bytetrack_yolox_x_8xb4-80e_crowdhuman_mot17halftrain_'
    'test-mot17halfval.py'
]

test_dataloader = dict(
    dataset=dict(
        data_root='data/MOT17/',
        ann_file='annotations/test_cocoformat.json',
        data_prefix=dict(img_path='test')))
