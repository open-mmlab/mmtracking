from mmdet.datasets import DATASETS

from .coco_video_dataset import CocoVideoDataset
from .parsers import CocoVID


@DATASETS.register_module()
class ImagenetVIDDataset(CocoVideoDataset):

    CLASSES = ('airplane', 'antelope', 'bear', 'bicycle', 'bird', 'bus', 'car',
               'cattle', 'dog', 'domestic_cat', 'elephant', 'fox',
               'giant_panda', 'hamster', 'horse', 'lion', 'lizard', 'monkey',
               'motorcycle', 'rabbit', 'red_panda', 'sheep', 'snake',
               'squirrel', 'tiger', 'train', 'turtle', 'watercraft', 'whale',
               'zebra')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_video_anns(self, ann_file):
        self.coco = CocoVID(ann_file)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}

        data_infos = []
        self.vid_ids = self.coco.get_vid_ids()
        self.img_ids = []
        for vid_id in self.vid_ids:
            img_ids = self.coco.get_img_ids_from_vid(vid_id)
            for img_id in img_ids:
                info = self.coco.load_imgs([img_id])[0]
                info['filename'] = info['file_name']
                if self.test_mode:
                    assert not info['is_train_frame'], \
                        'is_train_frame must be False in testing'
                    self.img_ids.append(img_id)
                    data_infos.append(info)
                elif info['is_train_frame']:
                    self.img_ids.append(img_id)
                    data_infos.append(info)
        return data_infos
