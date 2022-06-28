_base_ = ['./selsa_faster_rcnn_r50_dc5_7e_imagenetvid.py']
model = dict(
    detector=dict(
        backbone=dict(
            depth=101,
            init_cfg=dict(
                type='Pretrained', checkpoint='torchvision://resnet101'))))
