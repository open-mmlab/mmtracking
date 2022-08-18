_base_ = ['./fgfa_faster_rcnn-resnet50-dc5_8x1bs-7e_imagenetvid.py']
model = dict(
    detector=dict(
        backbone=dict(
            depth=101,
            init_cfg=dict(
                type='Pretrained', checkpoint='torchvision://resnet101'))))
