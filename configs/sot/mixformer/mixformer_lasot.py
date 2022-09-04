_base_ = ['./mixformer_got10k.py']

# model setting
model = dict(
    test_cfg=dict(
        search_factor=4.55,
        search_size=320,
        template_factor=2.0,
        template_size=128,
        update_interval=[200],
        online_size=[2],
        max_score_decay=[1.0],
    ))

data_root = 'data/'
data = dict(
    test=dict(
        type='LaSOTDataset',
        ann_file=data_root + 'lasot/annotations/lasot_test_infos.txt',
        img_prefix=data_root + 'lasot/LaSOTBenchmark'))
