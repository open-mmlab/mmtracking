from mmdet.datasets import DATASETS

from .coco_video_dataset import CocoVideoDataset


@DATASETS.register_module()
class MOT17Dataset(CocoVideoDataset):

    CLASSES = ('pedestrian')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def format_track_results(self, results, **kwargs):
        pass
