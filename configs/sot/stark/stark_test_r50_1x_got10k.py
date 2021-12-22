_base_ = ['./stark_st1_r50_1x_got10k.py']

# model setting
model = dict(
    type='Stark',
    head=dict(
        type='StarkHead',
        run_bbox_head=True,
        run_cls_head=True))