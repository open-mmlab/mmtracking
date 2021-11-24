_base_ = ['./tracktor_faster-rcnn_r50_fpn_4e_mot17-private-half.py']

model = dict(
    detector=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://download.openmmlab.com/mmtracking/fp16/faster-rcnn_r50_fpn_fp16_4e_mot17-half_20210730_002436-f4ba7d61.pth'  # noqa: E501
        )),
    reid=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://download.openmmlab.com/mmtracking/fp16/reid_r50_fp16_8x32_6e_mot17_20210731_033055-4747ee95.pth'  # noqa: E501
        )))
fp16 = dict(loss_scale=512.)
