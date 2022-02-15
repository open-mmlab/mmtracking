_base_ = ['./siamese_rpn_r50_20e_lasot.py']
optimizer_config = dict(type='SiameseRPNFp16OptimizerHook')
fp16 = dict(loss_scale=512.)
