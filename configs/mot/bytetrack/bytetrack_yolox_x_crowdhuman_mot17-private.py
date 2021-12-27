_base_ = ['./bytetrack_yolox_x_crowdhuman_mot17-private-half.py']

data = dict(
    test=dict(
        ann_file='data/MOT17/annotations/test_cocoformat.json',
        img_prefix='data/MOT17/test'))
