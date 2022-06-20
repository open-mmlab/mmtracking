_base_ = ['./bytetrack_yolox_x_crowdhuman_mot17-private-half.py']

test_dataloader = dict(
    dataset=dict(
        data_root='data/MOT17/',
        ann_file='annotations/test_cocoformat.json',
        data_prefix=dict(img_path='test')))
