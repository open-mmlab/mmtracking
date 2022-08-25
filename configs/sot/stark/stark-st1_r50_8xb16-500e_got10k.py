_base_ = [
    '../../_base_/datasets/got10k.py',
    './stark-st1_r50_8xb16-500e_got10k-lasot-trackingnet-coco_base.py'
]

train_pipeline = {{_base_.train_pipeline}}

# dataset settings
data_root = {{_base_.data_root}}
train_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type='GOT10kDataset',
        data_root=data_root,
        ann_file='GOT10k/annotations/got10k_train_infos.txt',
        data_prefix=dict(img_path='GOT10k'),
        pipeline=train_pipeline,
        test_mode=False))

# evaluator
val_evaluator = dict(outfile_prefix='results/stark_st1_got10k')
test_evaluator = val_evaluator
