_base_ = [
    './strongsort_yolox_x_8x4bs-80e_crowdhuman-mot17halftrain'
    '_test-mot17halfval.py'
]

# dataloader
val_dataloader = dict(
    dataset=dict(ann_file='annotations/train_cocoformat.json'))
test_dataloader = dict(
    dataset=dict(
        ann_file='annotations/test_cocoformat.json',
        data_prefix=dict(img_path='test')))

test_evaluator = dict(format_only=True)
