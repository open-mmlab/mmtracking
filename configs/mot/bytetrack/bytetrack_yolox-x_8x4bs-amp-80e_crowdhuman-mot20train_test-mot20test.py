_base_ = [
    './bytetrack_yolox-x_8x4bs-80e_crowdhuman-mot20train_test-mot20test.py'
]

# fp16 settings
optim_wrapper = dict(type='AmpOptimWrapper', loss_scale='dynamic')
test_cfg = dict(type='TestLoop', fp16=True)
