import random
from collections import defaultdict
from time import time

import mmcv
from mmcv.utils import get_logger


class mmVID(object):
    """API for instance annotations in videos.

    The annotation format is shown as follows.

    {
        'data': [
            {
                'name': str,
                'width': int,
                'height': int,
                'length': int,
                'fps': int,
                'images': [
                    {
                        'name': str,
                        'frame_id': int,
                        'annotations': [
                            {
                                'bbox': [x1, y1, x2, y2],
                                'label': int,
                                'instance_id': int,
                                'ignore': bool,
                                'crowd': bool,
                                'occluded': bool,
                                'truncated': bool
                            },
                            ...
                        ]
                    },
                    ...
                ]
            },
            ...
        ],
        'classes': [],
        'metas': {}
    }
    """

    def __init__(self, ann_file):
        self.logger = get_logger(name='mmVID')
        self.time = time()

        if isinstance(ann_file, str):
            dataset = mmcv.load(ann_file)
            self.logger.info(f'Loading annotations from {ann_file}.')
        elif isinstance(ann_file, dict):
            dataset = ann_file
            self.logger.info('Loading annotations.')
        else:
            raise TypeError('Input must be a file or Dictionary object.')

        self.logger.info(f'Loading done ({(time() - self.time):.2f}s)!')

        self._parse_data(dataset)

    def _parse_data(self, dataset):
        self.logger.info('Parsing annotations.')
        self.time = time()

        self.images = []
        self.label2imgs = defaultdict(list)
        self.ins2vid = defaultdict(list)
        self.vid2ins = defaultdict(list)

        num_imgs = 0
        num_anns = 0
        for vid_id, video in enumerate(dataset['data']):
            assert video['length'] == len(
                video['images']), 'Mismatched number of images.'

            video['images'] = sorted(
                video['images'], key=lambda x: x['frame_id'])

            for k, image in enumerate(video['images']):
                image['vid_id'] = vid_id
                self.images.append(image)

                num_anns += len(image['annotations'])
                anns = self.parse_anns(image['annotations'])

                labels = set(anns['labels'])
                for label in labels:
                    self.label2imgs[label].append(num_imgs + k)

                ins_ids = set(anns['instance_ids'])
                for ins_id in ins_ids:
                    if ins_id not in self.ins2vid.keys():
                        self.ins2vid[ins_id] = vid_id
                    if ins_id not in self.vid2ins[vid_id]:
                        self.vid2ins[vid_id].append(ins_id)

            num_imgs += video['length']

        self.videos = dataset['data']
        self.classes = dataset['classes']
        self.metas = dataset['metas']

        self.metas['statistics'] = dict(
            num_vids=len(self.videos),
            num_imgs=len(self.images),
            num_cls=len(self.classes),
            num_anns=num_anns,
            num_ins=len(self.ins2vid.keys()))
        self.logger.info(f'Parsing done ({(time() - self.time):.2f}s)!')
        self.logger.info(
            (f'The dataset contains '
             f"{self.metas['statistics']['num_cls']} classes, "
             f"{self.metas['statistics']['num_vids']} videos, "
             f"{self.metas['statistics']['num_imgs']} images, "
             f"{self.metas['statistics']['num_ins']} instances with "
             f"{self.metas['statistics']['num_anns']} boxes."))

    def filter_(self, img_ids, classes):
        # self._parse_data(img_ids, classes)
        # TODO: implement this function considering real cases
        pass

    def cls2label(self, classes):
        if isinstance(classes, list):
            for i, cls in enumerate(classes):
                classes[i] = self.classes.index(cls)
            return classes
        elif isinstance(classes, str):
            return self.classes.index(classes)
        else:
            raise TypeError('Input must be a string or a list of strings.')

    def get_vid(self, idx):
        return self.videos[idx]

    def get_img(self, idx):
        return self.images[idx]

    def get_imgs_by_cls(self, classes=None):
        if isinstance(classes, str) or isinstance(classes, int):
            classes = [classes]
        else:
            if not isinstance(classes, list):
                raise TypeError('Input should be str or int.')

        if isinstance(classes[0], str):
            labels = self.cls2label(classes)

        img_infos = []
        for label in labels:
            img_infos.extend(
                [self.images[idx] for idx in self.label2imgs[label]])

        return img_infos

    def get_anns(self, idx, parse=True, **kwargs):
        anns = self.get_img(idx)['annotations']
        if len(kwargs.keys()) > 0:
            assert parse, 'Only support options when parsing annotations.'
        if parse:
            anns = self.parse_anns(anns, **kwargs)
        return anns

    def parse_anns(self, raw_anns, classes=None, ins_ids=None):
        maps = dict(
            bboxes='bbox',
            labels='label',
            instance_ids='instance_id',
            crowd='crowd',
            ignore='ignore',
            occluded='occluded',
            truncated='truncated')

        if classes is not None:
            if isinstance(classes, str) or isinstance(classes, int):
                classes = [classes]
            else:
                if not isinstance(classes, list):
                    raise TypeError('Input should be str or int.')
            if isinstance(classes[0], str):
                classes = self.cls2label(classes)

        if ins_ids is not None:
            if isinstance(classes, int):
                ins_ids = [ins_ids]
            else:
                if not isinstance(classes, list):
                    raise TypeError('Input should be str or int.')

        anns = {k: [] for k in maps.keys()}
        for i, ann in enumerate(raw_anns):
            if classes is not None and ann['label'] not in classes:
                continue
            if ins_ids is not None and ann['instance_id'] not in ins_ids:
                continue
            for k, v in maps.items():
                if v not in ann.keys():
                    continue
                anns[k].append(ann[v])

        return anns

    def get_ins_ids(self, vid_id=None):
        if vid_id is None:
            return list(self.ins2vid.keys())
        else:
            return self.vid2ins[vid_id]

    def get_neighbor_imgs(self, idx, scope=-1):
        img_info = self.get_img(idx)
        ref_img_infos = self.get_vid(img_info['vid_id'])['images']

        frame_id = img_info['frame_id']
        front_imgs = ref_img_infos[:frame_id]
        behind_imgs = ref_img_infos[frame_id + 1:]

        if scope > 0:
            front_imgs = front_imgs[::-1][:scope][::-1]
            behind_imgs = behind_imgs[:scope]

        return front_imgs, behind_imgs

    def get_ref_img(self, idx, scope=-1, method='uniform'):
        front_imgs, behind_imgs = self.get_neighbor_imgs(idx, scope)

        if method == 'uniform':
            return random.choice(front_imgs + behind_imgs)
        else:
            raise NotImplementedError
