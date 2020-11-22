from collections import defaultdict

import numpy as np
from pycocotools.coco import COCO, _isArrayLike


class CocoVID(COCO):

    def __init__(self, annotation_file=None):
        assert annotation_file, 'Annotation file must be provided.'
        super(CocoVID, self).__init__(annotation_file=annotation_file)

    def createIndex(self):
        print('creating index...')
        anns, cats, imgs, vids = {}, {}, {}, {}
        imgToAnns, catToImgs, vidToImgs = defaultdict(list), defaultdict(
            list), defaultdict(list)

        if 'videos' in self.dataset:
            for video in self.dataset['videos']:
                vids[video['id']] = video

        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                anns[ann['id']] = ann

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                vidToImgs[img['video_id']].append(img)
                imgs[img['id']] = img

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                catToImgs[ann['category_id']].append(ann['image_id'])

        print('index created!')

        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats
        self.videos = vids
        self.vidToImgs = vidToImgs

    def get_vid_ids(self, vidIds=[]):
        vidIds = vidIds if _isArrayLike(vidIds) else [vidIds]

        if len(vidIds) == 0:
            ids = self.videos.keys()
        else:
            ids = set(vidIds)

        return list(ids)

    def get_img_ids_from_vid(self, vidId):
        img_infos = self.vidToImgs[vidId]
        ids = list(np.zeros([len(img_infos)], dtype=np.int))
        for img_info in img_infos:
            ids[img_info['frame_id']] = img_info['id']
        return ids

    def load_vids(self, ids=[]):
        if _isArrayLike(ids):
            return [self.videos[id] for id in ids]
        elif type(ids) == int:
            return [self.videos[ids]]
