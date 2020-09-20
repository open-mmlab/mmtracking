from mmdet.datasets import DATASETS

from .coco_video_dataset import CocoVideoDataset
from .parsers import CocoVID


@DATASETS.register_module()
class ImagenetVIDVideoDataset(CocoVideoDataset):

    CLASSES = ('airplane', 'antelope', 'bear', 'bicycle', 'bird', 'bus', 'car',
               'cattle', 'dog', 'domestic_cat', 'elephant', 'fox',
               'giant_panda', 'hamster', 'horse', 'lion', 'lizard', 'monkey',
               'motorcycle', 'rabbit', 'red_panda', 'sheep', 'snake',
               'squirrel', 'tiger', 'train', 'turtle', 'watercraft', 'whale',
               'zebra')

    def __init__(self, dff_mode=True, fgfa_mode=False, *args, **kwargs):
        self.dff_mode = dff_mode
        self.fgfa_mode = fgfa_mode
        assert self.dff_mode ^ self.fgfa_mode, 'Only support at most one mode'
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
                if self.test_mode is True and info['is_train_frame'] is False:
                    self.img_ids.append(img_id)
                    data_infos.append(info)
                elif self.test_mode is False \
                        and info['is_train_frame'] is True:
                    self.img_ids.append(img_id)
                    data_infos.append(info)
        return data_infos

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """
        num_ref_imgs = self.ref_img_sampler['num_ref_imgs']
        if self.dff_mode:
            num_ref_imgs = 1 if not self.load_as_video else num_ref_imgs
            assert num_ref_imgs == 1, 'only support 1 ref_img in dff mode'
            results = super().prepare_train_img(idx)
            results['is_video_data'] = self.load_as_video
            return results
        elif self.fgfa_mode:
            num_ref_imgs = 2 if not self.load_as_video else num_ref_imgs
            assert num_ref_imgs == 2, 'only support 2 ref_imgs in fgfa mode'
            frame_range = self.ref_img_sampler['frame_range']
            if isinstance(frame_range, int):
                assert frame_range > 0, 'frame_range must bigger than 0.'
                frame_range = [-frame_range, frame_range]
            elif isinstance(frame_range, list):
                assert len(frame_range) == 2, 'The length must be 2.'
                assert frame_range[0] < 0 and frame_range[1] > 0
                for i in frame_range:
                    assert isinstance(i, int), 'Each element must be int.'
            else:
                raise TypeError('The type of frame_range must be int or list.')

            img_info = self.data_infos[idx]
            results = self.prepare_results(img_info)
            self.pre_pipeline(results)
            all_results = []
            all_results.append(results)

            for i in range(num_ref_imgs):
                ref_img_sampler = self.ref_img_sampler.copy()
                ref_img_sampler['num_ref_imgs'] = 1
                if i == 0:
                    ref_img_sampler['frame_range'] = [frame_range[0], -1]
                else:
                    ref_img_sampler['frame_range'] = [1, frame_range[1]]

                ref_img_info = self.ref_img_sampling(img_info,
                                                     **ref_img_sampler)
                ref_results = self.prepare_results(ref_img_info)
                assert self.match_gts is False, \
                    'matching gts is not supported in fgfa mode for now'
                self.pre_pipeline(ref_results)
                all_results.append(ref_results)
            all_results = self.pipeline(all_results)
            all_results['is_video_data'] = self.load_as_video
            return all_results
