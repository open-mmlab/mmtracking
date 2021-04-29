_base_ = ['./deepsort_faster-rcnn_fpn_4e_mot17-private-half.py']
model = dict(
    pretrains=dict(
        detector=  # noqa: E251
        'work_dirs/detector/faster-rcnn_r50_fpn_4e_mot15-half/latest.pth',  # noqa: E501
        reid=  # noqa: E251
        'work_dirs/reid/mot15/myself_best_reid_mot15.pth'  # noqa: E501
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