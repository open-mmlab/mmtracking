_base_ = ['./tracktor_faster-rcnn_r50_fpn_4e_mot17-public.py']
model = dict(
    pretrains=dict(
        detector=  # noqa: E251
        # 'https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot15-9e00ac7f.pth',  # noqa: E501
        '/mnt/lustre/share/shensanjing/model/sort/publish_model/faster-rcnn_r50_fpn_4e_mot15-9e00ac7f.pth',
        reid=  # noqa: E251
        # 'https://download.openmmlab.com/mmtracking/mot/reid/tracktor_reid_r50_69e_mot15-f7980743.pth'  # noqa: E501
        '/mnt/lustre/share/shensanjing/model/sort/publish_model/tracktor_reid_r50_69e_mot15-f7980743.pth'
    ))
data_root = 'data/MOT15/'
test_set = 'train'
data = dict(
    train=dict(
        ann_file=data_root + 'annotations/train_cocoformat.json',
        img_prefix=data_root + 'train'),
    val=dict(
        ann_file=data_root + 'annotations/train_cocoformat.json',
        img_prefix=data_root + 'train',
        detection_file=data_root + 'annotations/train_detections.pkl'),
    test=dict(
        ann_file=data_root + f'annotations/{test_set}_cocoformat.json',
        img_prefix=data_root + test_set,
        detection_file=data_root + f'annotations/{test_set}_detections.pkl'))
