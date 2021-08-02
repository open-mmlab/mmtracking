_base_ = [
    '../mot/tracktor/tracktor_faster-rcnn_r50_fpn_4e_mot17-private-half.py'
]
model = dict(
    pretrains=dict(
        detector=  # noqa: E251
        '/mnt/lustre/gongtao.vendor/Codes/tracking/pt1.5s1_mmtrack_sot/mmtrack_output/faster-rcnn_r50_fpn_fp16_4e_mot17-half/epoch_4.pth',  # noqa: E501
        reid=  # noqa: E251
        '/mnt/lustre/gongtao.vendor/Codes/tracking/pt1.5s1_mmtrack_sot/mmtrack_output/reid_r50_fp16_8x32_6e_mot17/epoch_6.pth'  # noqa: E501
    ))
fp16 = dict(loss_scale=512.)
