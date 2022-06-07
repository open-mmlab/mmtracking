# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmtrack.datasets.transforms import PairSampling, TridentSampling


class TestPairSampling:

    def setup_class(cls):
        num_frames = 60
        pair_video_infos = []
        filenames = ['{:08d}.jpg'.format(i) for i in range(num_frames)]
        frame_ids = np.arange(num_frames)
        bboxes = np.ones((num_frames, 4))
        bboxes_isvalid = np.ones(num_frames, dtype=bool)
        visible = bboxes_isvalid.copy()

        for video_id in range(2):
            video_info = dict(
                bboxes=bboxes,
                bboxes_isvalid=bboxes_isvalid,
                visible=visible,
                img_paths=filenames,
                frame_ids=frame_ids,
                video_id=video_id,
                video_length=60)
            pair_video_infos.append(video_info)

        cls.num_frames = num_frames
        cls.pair_video_infos = pair_video_infos
        cls.pair_sampling = PairSampling(
            frame_range=5, pos_prob=0.8, filter_template_img=False)

    def test_transform(self):
        results = self.pair_sampling(self.pair_video_infos)

        frame_ids = results['frame_id']
        assert len(frame_ids) == 2
        for frame_id in frame_ids:
            assert 0 <= frame_id < self.num_frames

        instances = results['instances']
        assert len(instances) == 2
        for instance in instances:
            assert len(instance) == 1
            assert (instance[0]['bbox'] == np.ones(4)).all()
            assert (instance[0]['bbox_label'] <= np.ones(1)).all()


class TestTridentSampling:

    def setup_class(cls):
        num_frames = 60
        pair_video_infos = []
        filenames = ['{:08d}.jpg'.format(i) for i in range(num_frames)]
        frame_ids = np.arange(num_frames)
        bboxes = np.ones((num_frames, 4))
        bboxes_isvalid = np.ones(num_frames, dtype=bool)
        visible = bboxes_isvalid.copy()

        for video_id in range(2):
            bboxes_isvalid = np.ones(num_frames, dtype=bool)
            random_invalid_index = np.random.randint(0, num_frames, 4)
            bboxes_isvalid[random_invalid_index] = False
            visible = bboxes_isvalid.copy()
            random_invalid_index = np.random.randint(0, num_frames, 4)
            visible[random_invalid_index] = False
            video_info = dict(
                bboxes=bboxes,
                bboxes_isvalid=bboxes_isvalid,
                visible=visible,
                img_paths=filenames,
                frame_ids=frame_ids,
                video_id=video_id,
                video_length=60)
            pair_video_infos.append(video_info)

        cls.num_frames = num_frames
        cls.pair_video_infos = pair_video_infos
        cls.pair_sampling = TridentSampling(
            num_search_frames=1,
            num_template_frames=2,
            max_frame_range=[200],
            cls_pos_prob=0.5,
            train_cls_head=True)

    def test_transform(self):
        results = self.pair_sampling(self.pair_video_infos)

        frame_ids = results['frame_id']
        assert len(frame_ids) == 3
        for frame_id in frame_ids:
            assert 0 <= frame_id < self.num_frames

        instances = results['instances']
        assert len(instances) == 3
        for instance in instances:
            assert len(instance) == 1
            assert (instance[0]['bbox'] == np.ones(4)).all()
            assert (instance[0]['bbox_label'] <= np.ones(1)).all()
