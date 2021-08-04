_base_ = ['../sot/siamese_rpn/siamese_rpn_r50_1x_lasot.py']
optimizer_config = dict(type='SiameseRPNFp16OptimizerHook')
fp16 = dict(loss_scale=512.)
