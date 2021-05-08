_base_ = ['./sort_faster-rcnn_fpn_4e_mot17-private-half.py']
model = dict(
    pretrains=dict(
        detector=  # noqa: E251
        # 'https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot15-half-f48f6578.pth'  # noqa: E501
        '/mnt/lustre/share_data/shensanjing/model/sort/publish_model/faster-rcnn_r50_fpn_4e_mot15-half-f48f6578.pth'
    ))
data_root = 'data/MOT15/'
data = dict(
    train=dict(
        ann_file=data_root + 'annotations/half-train_cocoformat.json',
        img_prefix=data_root + 'train'),
    val=dict(
        ann_file=data_root + 'annotations/half-val_cocoformat.json',
        img_prefix=data_root + 'train'),
    test=dict(
        ann_file=data_root + 'annotations/half-val_cocoformat.json',
        img_prefix=data_root + 'train'))
