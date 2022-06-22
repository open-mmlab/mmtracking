_base_ = ['./sort_faster-rcnn_fpn_4e_mot17-private-half.py']
model = dict(
    detector=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-ffa52ae7.pth'  # noqa: E501
        )))

data_root = 'data/MOT17/'
val_dataloader = dict(
    dataset=dict(ann_file=data_root + 'annotations/train_cocoformat.json'))
test_dataloader = val_dataloader
