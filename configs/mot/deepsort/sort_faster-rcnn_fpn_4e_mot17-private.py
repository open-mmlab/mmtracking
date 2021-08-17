_base_ = ['./sort_faster-rcnn_fpn_4e_mot17-private-half.py']
model = dict(
    detector=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-ffa52ae7.pth'  # noqa: E501
        )))
data_root = 'data/MOT17/'
test_set = 'train'
data = dict(
    train=dict(ann_file=data_root + 'annotations/train_cocoformat.json'),
    val=dict(ann_file=data_root + 'annotations/train_cocoformat.json'),
    test=dict(
        ann_file=data_root + f'annotations/{test_set}_cocoformat.json',
        img_prefix=data_root + test_set))
