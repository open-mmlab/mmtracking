_base_ = ['./tracktor_faster-rcnn_fpn_4e_mot17-private.py']
model = dict(
    pretrains=dict(
        detector=  # noqa: E251
        # 'https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot16-ccb2ff52.pth',  # noqa: E501
        '/mnt/lustre/share/shensanjing/model/sort/publish_model/faster-rcnn_r50_fpn_4e_mot16-ccb2ff52.pth',
        reid=  # noqa: E251
        # 'https://download.openmmlab.com/mmtracking/mot/reid/tracktor_reid_r50_69e_mot16-a2e459b3.pth'  # noqa: E501
        '/mnt/lustre/share/shensanjing/model/sort/publish_model/tracktor_reid_r50_69e_mot16-a2e459b3.pth'
    ))
data_root = 'data/MOT16/'
test_set = 'train'
data = dict(
    train=dict(
        ann_file=data_root + 'annotations/train_cocoformat.json',
        img_prefix=data_root + 'train'),
    val=dict(
        ann_file=data_root + 'annotations/train_cocoformat.json',
        img_prefix=data_root + 'train'),
    test=dict(
        ann_file=data_root + f'annotations/{test_set}_cocoformat.json',
        img_prefix=data_root + test_set))
