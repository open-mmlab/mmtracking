_base_ = ['./mixformer_cvt_500e_got10k.py']

# model setting
model = dict(
    test_cfg=dict(
        search_factor=4.5,
        search_size=320,
        template_factor=2.0,
        template_size=128,
        update_interval=[25],
        online_size=[1],
        max_score_decay=[1.0],
    ))

data_root = 'data/'
data = dict(
    test=dict(
        type='TrackingNetDataset',
        ann_file=data_root +
        'trackingnet/annotations/trackingnet_test_infos.txt',
        img_prefix=data_root + 'trackingnet'))
