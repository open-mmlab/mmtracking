_base_ = [
    './masktrack-rcnn_mask-rcnn-resnet50-fpn_8x1bs-12e_youtubevis2019.py'
]

data_root = 'data/youtube_vis_2021/'
dataset_version = data_root[-5:-1]

# dataloader
train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        dataset_version=dataset_version,
        ann_file='annotations/youtube_vis_2021_train.json'))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        dataset_version=dataset_version,
        ann_file='annotations/youtube_vis_2021_valid.json'))
test_dataloader = val_dataloader
