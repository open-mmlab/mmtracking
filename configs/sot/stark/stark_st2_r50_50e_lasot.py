_base_ = ['../../_base_/datasets/lasot.py', './stark_st2_r50_50e_base.py']

# model setting
model = dict(test_cfg=dict(update_intervals=[300]))
