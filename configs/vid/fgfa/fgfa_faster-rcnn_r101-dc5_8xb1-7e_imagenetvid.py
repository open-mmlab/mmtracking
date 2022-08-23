_base_ = ['./fgfa_faster-rcnn_r50-dc5_8xb1-7e_imagenetvid.py']
model = dict(
    detector=dict(
        backbone=dict(
            depth=101,
            init_cfg=dict(
                type='Pretrained', checkpoint='torchvision://resnet101'))))
