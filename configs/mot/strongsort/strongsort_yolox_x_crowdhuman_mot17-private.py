_base_ = ['./strongsort_yolox_x_crowdhuman_mot17-private-half.py']

# dataloader
val_dataloader = dict(
    dataset=dict(ann_file='annotations/train_cocoformat.json')
)
test_dataloader = dict(
    dataset=dict(
        ann_file='annotations/test_cocoformat.json',
        data_prefix=dict(img_path='test'),
))

test_evaluator = dict(
    format_only=True,
    resfile_path='/data1/dyh/results/mmtracking/strongsort++_mot17-test'  #
)
