_base_ = ['./mixformer_got10k.py']

data_root = 'data/'
data = dict(
    test=dict(
        type='LaSOTDataset',
        ann_file=data_root + 'lasot/annotations/lasot_test_infos.txt',
        img_prefix=data_root + 'lasot/LaSOTBenchmark'))
