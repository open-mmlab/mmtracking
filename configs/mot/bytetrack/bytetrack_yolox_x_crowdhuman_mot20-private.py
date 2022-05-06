_base_ = ['./bytetrack_yolox_x_crowdhuman_mot20-private-half.py']

data = dict(
    train=dict(
        dataset=dict(
            ann_file=[
                'data/MOT20/annotations/train_cocoformat.json',
                'data/crowdhuman/annotations/crowdhuman_train.json',
                'data/crowdhuman/annotations/crowdhuman_val.json'
            ],
        )
    ),
    val=dict(
        ann_file='data/MOT17/annotations/train_cocoformat.json',
        img_prefix='data/MOT17/train'
    ),
    test=dict(
        ann_file='data/MOT20/annotations/test_cocoformat.json',
        img_prefix='data/MOT20/test'
    )
)
evaluation = dict(metric=['bbox', 'track'], interval=1)