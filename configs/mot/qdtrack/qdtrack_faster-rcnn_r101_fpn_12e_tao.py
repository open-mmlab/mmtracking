# model settings
_base_ = ['./qdtrack_faster-rcnn_r101_fpn_24e_lvis.py']
model = dict(freeze_detector=True)
data_root = 'data/tao/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        dataset=dict(
            classes=data_root + 'annotations/tao_classes.txt',
            ann_file=data_root + 'annotations/qdtrack_482/train_482_ours.json',
            img_prefix=data_root + 'frames/',
            load_as_video=True,
            key_img_sampler=dict(interval=1),
            ref_img_sampler=dict(
                num_ref_imgs=1, frame_range=[-1, 1], method='uniform'))))
# learning policy
lr_config = dict(step=[8, 11])
total_epochs = 12
load_from = None
resume_from = None
evaluation = dict(metric=['track'], start=8, interval=1)
work_dir = None
