_base_ = ['./bytetrack_yolox_x_crowdhuman_mot20-private.py']

# fp16 settings
optim_wrapper = dict(type='AmpOptimWrapper', loss_scale='dynamic')
test_cfg = dict(type='TestLoop', fp16=True)
